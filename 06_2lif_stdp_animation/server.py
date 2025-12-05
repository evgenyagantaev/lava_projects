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

from backend import simulate_stdp, SimulationState


async def lif_stream(
    rate: float,
    dv: float,
    threshold: float,
    spike_amp: float,
    delay_ms: int,
    chunk_steps: int,
    w_init: float,
    w_min: float,
    w_max: float,
    decay_rate: float,
    w_baseline: float,
) -> AsyncIterator[Dict[str, float]]:
    """
    Generate frames for two LIF neurons + STDP synapse in chunks.

    Uses persistent state for continuous simulation across chunks.
    """
    t = 0
    seed = 0

    # Create persistent state for this connection
    state = SimulationState(
        weight=w_init * threshold,
        w_min=w_min * threshold,
        w_max=w_max * threshold,
    )

    while True:
        traces = await asyncio.to_thread(
            simulate_stdp,
            num_steps=chunk_steps,
            rate=rate,
            threshold=threshold,
            spike_fraction=spike_amp / threshold,
            dv=dv,
            seed=seed,
            w_init=w_init,
            w_min=w_min,
            w_max=w_max,
            decay_rate=decay_rate,
            w_baseline=w_baseline,
            use_continuous_state=False,  # Use explicit state instead
            state=state,  # Pass state for continuity
        )
        seed += 1

        neurons = traces["neurons"]
        lengths = [
            len(neurons[0]["membrane_potential"]),
            len(neurons[1]["membrane_potential"]),
            len(neurons[0]["spikes"]),
            len(neurons[1]["spikes"]),
            len(neurons[0]["input_any"]),
            len(neurons[1]["input_any"]),
            len(traces["inputs_detail"][0][0]),
            len(traces["inputs_detail"][1][0]),
            len(traces["stdp"]["pre_trace"]),
            len(traces["stdp"]["post_trace"]),
            len(traces["stdp"]["weight"]),
        ]
        steps = min(chunk_steps, *lengths)
        if steps <= 0:
            continue

        for i in range(steps):
            yield {
                "t": t,
                "threshold": threshold,
                "delay_ms": delay_ms,
                "v": [neurons[0]["membrane_potential"][i], neurons[1]["membrane_potential"][i]],
                "spike": [neurons[0]["spikes"][i], neurons[1]["spikes"][i]],
                "input": [neurons[0]["input_any"][i], neurons[1]["input_any"][i]],
                "input_detail": [
                    [traces["inputs_detail"][0][0][i], traces["inputs_detail"][0][1][i], traces["inputs_detail"][0][2][i]],
                    [traces["inputs_detail"][1][0][i], traces["inputs_detail"][1][1][i], traces["inputs_detail"][1][2][i]],
                ],
                "pre_trace": traces["stdp"]["pre_trace"][i],
                "post_trace": traces["stdp"]["post_trace"][i],
                "weight": traces["stdp"]["weight"][i],
                # Include weight bounds for UI
                "w_min": traces["stdp"]["w_min"],
                "w_max": traces["stdp"]["w_max"],
            }
            t += 1
            await asyncio.sleep(delay_ms / 1000.0)


async def handler(
    websocket,
    *,
    rate: float,
    dv: float,
    threshold: float,
    spike_amp: float,
    delay_ms: int,
    chunk_steps: int,
    w_init: float,
    w_min: float,
    w_max: float,
    decay_rate: float,
    w_baseline: float,
):
    """Handle WebSocket connection with per-connection state."""
    stream = lif_stream(
        rate=rate,
        dv=dv,
        threshold=threshold,
        spike_amp=spike_amp,
        delay_ms=delay_ms,
        chunk_steps=chunk_steps,
        w_init=w_init,
        w_min=w_min,
        w_max=w_max,
        decay_rate=decay_rate,
        w_baseline=w_baseline,
    )
    async for frame in stream:
        await websocket.send(json.dumps(frame))


async def main():
    parser = argparse.ArgumentParser(description="Realtime LIF+STDP WebSocket streamer with state continuity")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--rate", type=float, default=0.05, help="Input spike probability per tick")
    parser.add_argument("--dv", type=float, default=0.04, help="Membrane leak factor")
    parser.add_argument("--threshold", type=float, default=1.0, help="Spike threshold")
    parser.add_argument("--spike-fraction", type=float, default=0.4, help="Input spike amplitude as fraction of threshold")
    parser.add_argument("--delay-ms", type=int, default=80, help="Delay between steps for visualization")
    parser.add_argument("--chunk-steps", type=int, default=256, help="Simulation steps per chunk")

    # Weight management parameters
    parser.add_argument("--w-init", type=float, default=0.2, help="Initial weight (fraction of threshold)")
    parser.add_argument("--w-min", type=float, default=0.0, help="Minimum weight (fraction of threshold)")
    parser.add_argument("--w-max", type=float, default=1.0, help="Maximum weight (fraction of threshold)")
    parser.add_argument("--decay-rate", type=float, default=0.0, help="Weight decay rate towards baseline (0 = disabled)")
    parser.add_argument("--w-baseline", type=float, default=0.1, help="Weight decay baseline (fraction of threshold)")

    args = parser.parse_args()

    spike_amp = args.threshold * args.spike_fraction

    async def _handler(ws):
        return await handler(
            ws,
            rate=args.rate,
            dv=args.dv,
            threshold=args.threshold,
            spike_amp=spike_amp,
            delay_ms=args.delay_ms,
            chunk_steps=args.chunk_steps,
            w_init=args.w_init,
            w_min=args.w_min,
            w_max=args.w_max,
            decay_rate=args.decay_rate,
            w_baseline=args.w_baseline,
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
        f"Starting LIF+STDP stream on ws://{args.host}:{args.port}/stream\n"
        f"  Parameters: rate={args.rate}, dv={args.dv}, spike_amp={spike_amp:.2f}\n"
        f"  Weight: init={args.w_init}, min={args.w_min}, max={args.w_max}, decay={args.decay_rate}, baseline={args.w_baseline}\n"
        f"  Chunks: {args.chunk_steps} steps, delay={args.delay_ms}ms"
    )

    async with websockets.serve(_handler, args.host, args.port, ping_interval=None):
        await stop.wait()


if __name__ == "__main__":
    asyncio.run(main())
