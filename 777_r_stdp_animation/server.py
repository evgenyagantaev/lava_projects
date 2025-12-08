"""Realtime LIF+STDP stream server backed by lava-nc (two connected neurons).

Runs a WebSocket server that streams traces of two floating-point LIF neurons
connected by a plastic STDP synapse (neuron0 -> neuron1). Frames are generated
in chunks and emitted in realtime.

Features:
  - Continuous state between chunks (weights, membrane potentials, traces)
  - Weight clipping and decay
  - WebSocket streaming for real-time visualization

Run:
  python server.py --host 127.0.0.1 --port 8765
"""

import argparse
import asyncio
import json
import signal
import sys
from pathlib import Path
from typing import AsyncIterator, Dict

import websockets

# Prefer installed lava-nc over local source tree
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    p_str = str(p)
    if p_str in sys.path:
        sys.path.remove(p_str)

import numpy as np

from backend import simulate_rstdp, RSimulationState


async def lif_stream(
    threshold: float,
    delay_ms: int,
    chunk_steps: int,
    w_init: float,
    w_min: float,
    w_max: float,
) -> AsyncIterator[Dict[str, float]]:
    """
    Generate frames for 1 pre-synaptic and 2 post-synaptic neurons with
    R-STDP plasticity in chunks. Uses persistent state for continuous
    simulation across chunks.
    """
    t = 0
    seed = 0

    # Create persistent R-STDP state for this connection
    state = RSimulationState(
        weights=np.array([w_init, w_init], dtype=float) * threshold,
        w_min=w_min * threshold,
        w_max=w_max * threshold,
    )

    while True:
        traces = await asyncio.to_thread(
            simulate_rstdp,
            num_steps=chunk_steps,
            threshold=threshold,
            w_init=w_init,
            w_min=w_min,
            w_max=w_max,
            seed=seed,
            use_continuous_state=False,
            state=state,
        )
        seed += 1

        neurons = traces["neurons"]
        rstdp = traces["rstdp"]
        lengths = [
            len(neurons[0]["membrane_potential"]),
            len(neurons[1]["membrane_potential"]),
            len(neurons[2]["membrane_potential"]),
            len(neurons[0]["spikes"]),
            len(neurons[1]["spikes"]),
            len(neurons[2]["spikes"]),
            len(neurons[0]["input_any"]),
            len(neurons[1]["input_any"]),
            len(neurons[2]["input_any"]),
            len(rstdp["pre_trace"]),
            len(rstdp["post_trace_A"]),
            len(rstdp["post_trace_B"]),
            len(rstdp["eligibility_A"]),
            len(rstdp["eligibility_B"]),
            len(rstdp["reward_A"]),
            len(rstdp["reward_B"]),
            len(rstdp["weight_A"]),
            len(rstdp["weight_B"]),
        ]
        steps = min(chunk_steps, *lengths)
        if steps <= 0:
            continue

        for i in range(steps):
            yield {
                "t": t,
                "threshold": threshold,
                "delay_ms": delay_ms,
                "v": [
                    neurons[0]["membrane_potential"][i],
                    neurons[1]["membrane_potential"][i],
                    neurons[2]["membrane_potential"][i],
                ],
                "spike": [
                    neurons[0]["spikes"][i],
                    neurons[1]["spikes"][i],
                    neurons[2]["spikes"][i],
                ],
                "input": [
                    neurons[0]["input_any"][i],
                    neurons[1]["input_any"][i],
                    neurons[2]["input_any"][i],
                ],
                "pre_trace": rstdp["pre_trace"][i],
                "post_trace": [rstdp["post_trace_A"][i], rstdp["post_trace_B"][i]],
                "eligibility": [rstdp["eligibility_A"][i], rstdp["eligibility_B"][i]],
                "reward": [rstdp["reward_A"][i], rstdp["reward_B"][i]],
                "weight": [rstdp["weight_A"][i], rstdp["weight_B"][i]],
                "w_min": rstdp["w_min"],
                "w_max": rstdp["w_max"],
            }
            t += 1
            await asyncio.sleep(delay_ms / 1000.0)


async def handler(
    websocket,
    *,
    threshold: float,
    delay_ms: int,
    chunk_steps: int,
    w_init: float,
    w_min: float,
    w_max: float,
):
    """Handle WebSocket connection with per-connection state."""
    stream = lif_stream(
        threshold=threshold,
        delay_ms=delay_ms,
        chunk_steps=chunk_steps,
        w_init=w_init,
        w_min=w_min,
        w_max=w_max,
    )
    async for frame in stream:
        await websocket.send(json.dumps(frame))


async def main():
    parser = argparse.ArgumentParser(description="Realtime LIF+R-STDP WebSocket streamer with state continuity")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--threshold", type=float, default=1.0, help="Spike threshold")
    parser.add_argument("--delay-ms", type=int, default=80, help="Delay between steps for visualization")
    parser.add_argument("--chunk-steps", type=int, default=256, help="Simulation steps per chunk")

    # Weight management parameters (normalized)
    parser.add_argument("--w-init", type=float, default=0.2, help="Initial weight (fraction of threshold)")
    parser.add_argument("--w-min", type=float, default=0.0, help="Minimum weight (fraction of threshold)")
    parser.add_argument("--w-max", type=float, default=1.0, help="Maximum weight (fraction of threshold)")

    args = parser.parse_args()

    async def _handler(ws):
        return await handler(
            ws,
            threshold=args.threshold,
            delay_ms=args.delay_ms,
            chunk_steps=args.chunk_steps,
            w_init=args.w_init,
            w_min=args.w_min,
            w_max=args.w_max,
        )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    # Graceful shutdown handling
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
    except NotImplementedError:
        # Windows fallback
        def _handle(_sig, _frame):
            stop.set()
        signal.signal(signal.SIGINT, _handle)

    print(
        f"Starting LIF+R-STDP stream on ws://{args.host}:{args.port}/stream\n"
        f"  Parameters: threshold={args.threshold}\n"
        f"  Weight: init={args.w_init}, min={args.w_min}, max={args.w_max}\n"
        f"  Chunks: {args.chunk_steps} steps, delay={args.delay_ms}ms"
    )

    async with websockets.serve(_handler, args.host, args.port, ping_interval=None):
        await stop.wait()


if __name__ == "__main__":
    asyncio.run(main())
