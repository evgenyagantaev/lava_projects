"""Realtime R-STDP stream server backed by lava-nc.

Runs a WebSocket server that streams traces of R-STDP network:
- 1 pre-synaptic LIF neuron
- 2 post-synaptic RSTDPLIF neurons (A and B)
- Plastic synapses with reward-modulated learning

Features:
  - Continuous state between chunks (weights, membrane potentials, traces)
  - Weight clipping
  - Eligibility and reward trace streaming
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

from backend import simulate_rstdp, SimulationState


async def rstdp_stream(
    rate_pre: float,
    rate_post_a: float,
    rate_post_b: float,
    dv: float,
    threshold: float,
    spike_amp: float,
    delay_ms: int,
    chunk_steps: int,
    learning_rate: float,
    pre_trace_tau: float,
    post_trace_tau: float,
    eligibility_tau: float,
    reward_prob: float,
    reward_min_len: int,
    reward_max_len: int,
    reward_amplitude: float,
    w_init: float,
    w_min: float,
    w_max: float,
) -> AsyncIterator[Dict[str, float]]:
    """
    Generate frames for R-STDP network (1 pre, 2 post) in chunks.

    Uses persistent state for continuous simulation across chunks.
    """
    t = 0
    seed = 0

    # Create persistent state for this connection
    state = SimulationState(
        w_A=w_init,
        w_B=w_init,
        w_min=w_min,
        w_max=w_max,
    )

    while True:
        traces = await asyncio.to_thread(
            simulate_rstdp,
            num_steps=chunk_steps,
            rate_pre=rate_pre,
            rate_post=(rate_post_a, rate_post_b),
            threshold=threshold,
            spike_fraction=spike_amp / threshold,
            dv=dv,
            seed=seed,
            learning_rate=learning_rate,
            pre_trace_tau=pre_trace_tau,
            post_trace_tau=post_trace_tau,
            eligibility_tau=eligibility_tau,
            reward_prob=reward_prob,
            reward_min_len=reward_min_len,
            reward_max_len=reward_max_len,
            reward_amplitude=reward_amplitude,
            w_init=w_init,
            w_min=w_min,
            w_max=w_max,
            use_continuous_state=False,
            state=state,
        )
        seed += 1

        neurons = traces["neurons"]
        rstdp = traces["rstdp"]

        # Get minimum length across all arrays
        lengths = [
            len(neurons[0]["membrane_potential"]),
            len(neurons[1]["membrane_potential"]),
            len(neurons[2]["membrane_potential"]),
            len(neurons[0]["spikes"]),
            len(neurons[1]["spikes"]),
            len(neurons[2]["spikes"]),
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
                # Neuron data (3 neurons: Pre, Post A, Post B)
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
                # R-STDP specific data
                "pre_trace": rstdp["pre_trace"][i],
                "post_trace": [
                    rstdp["post_trace_A"][i],
                    rstdp["post_trace_B"][i],
                ],
                "eligibility": [
                    rstdp["eligibility_A"][i],
                    rstdp["eligibility_B"][i],
                ],
                "reward": [
                    rstdp["reward_A"][i],
                    rstdp["reward_B"][i],
                ],
                "weight": [
                    rstdp["weight_A"][i],
                    rstdp["weight_B"][i],
                ],
                # Weight bounds for UI
                "w_min": rstdp["w_min"],
                "w_max": rstdp["w_max"],
            }
            t += 1
            await asyncio.sleep(delay_ms / 1000.0)


async def handler(
    websocket,
    *,
    rate_pre: float,
    rate_post_a: float,
    rate_post_b: float,
    dv: float,
    threshold: float,
    spike_amp: float,
    delay_ms: int,
    chunk_steps: int,
    learning_rate: float,
    pre_trace_tau: float,
    post_trace_tau: float,
    eligibility_tau: float,
    reward_prob: float,
    reward_min_len: int,
    reward_max_len: int,
    reward_amplitude: float,
    w_init: float,
    w_min: float,
    w_max: float,
):
    """Handle WebSocket connection with per-connection state."""
    stream = rstdp_stream(
        rate_pre=rate_pre,
        rate_post_a=rate_post_a,
        rate_post_b=rate_post_b,
        dv=dv,
        threshold=threshold,
        spike_amp=spike_amp,
        delay_ms=delay_ms,
        chunk_steps=chunk_steps,
        learning_rate=learning_rate,
        pre_trace_tau=pre_trace_tau,
        post_trace_tau=post_trace_tau,
        eligibility_tau=eligibility_tau,
        reward_prob=reward_prob,
        reward_min_len=reward_min_len,
        reward_max_len=reward_max_len,
        reward_amplitude=reward_amplitude,
        w_init=w_init,
        w_min=w_min,
        w_max=w_max,
    )
    async for frame in stream:
        await websocket.send(json.dumps(frame))


async def main():
    parser = argparse.ArgumentParser(description="Realtime R-STDP WebSocket streamer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)

    # Spike rates
    parser.add_argument("--rate-pre", type=float, default=0.05, help="Pre-synaptic spike probability")
    parser.add_argument("--rate-post-a", type=float, default=0.03, help="Post A spike probability")
    parser.add_argument("--rate-post-b", type=float, default=0.03, help="Post B spike probability")

    # Neuron parameters
    parser.add_argument("--dv", type=float, default=0.04, help="Membrane leak factor")
    parser.add_argument("--threshold", type=float, default=1.0, help="Spike threshold")
    parser.add_argument("--spike-fraction", type=float, default=0.4, help="Input spike amplitude as fraction of threshold")

    # Timing
    parser.add_argument("--delay-ms", type=int, default=50, help="Delay between steps for visualization")
    parser.add_argument("--chunk-steps", type=int, default=200, help="Simulation steps per chunk")

    # R-STDP parameters
    parser.add_argument("--learning-rate", type=float, default=0.1, help="R-STDP learning rate")
    parser.add_argument("--pre-trace-tau", type=float, default=10.0, help="Pre-synaptic trace decay tau")
    parser.add_argument("--post-trace-tau", type=float, default=10.0, help="Post-synaptic trace decay tau")
    parser.add_argument("--eligibility-tau", type=float, default=2.0, help="Eligibility trace decay tau")

    # Reward generation (random windows)
    parser.add_argument(
        "--reward-prob",
        type=float,
        default=0.02,
        help="Per-step probability to start a new reward window when inactive",
    )
    parser.add_argument(
        "--reward-min-len",
        type=int,
        default=7,
        help="Minimum reward window length (in simulation steps)",
    )
    parser.add_argument(
        "--reward-max-len",
        type=int,
        default=30,
        help="Maximum reward window length (in simulation steps)",
    )
    parser.add_argument(
        "--reward-amplitude",
        type=float,
        default=1.0,
        help="Reward signal amplitude",
    )

    # Weight management
    parser.add_argument("--w-init", type=float, default=0.5, help="Initial weight")
    parser.add_argument("--w-min", type=float, default=0.0, help="Minimum weight")
    parser.add_argument("--w-max", type=float, default=1.0, help="Maximum weight")

    args = parser.parse_args()

    spike_amp = args.threshold * args.spike_fraction

    async def _handler(ws):
        return await handler(
            ws,
            rate_pre=args.rate_pre,
            rate_post_a=args.rate_post_a,
            rate_post_b=args.rate_post_b,
            dv=args.dv,
            threshold=args.threshold,
            spike_amp=spike_amp,
            delay_ms=args.delay_ms,
            chunk_steps=args.chunk_steps,
            learning_rate=args.learning_rate,
            pre_trace_tau=args.pre_trace_tau,
            post_trace_tau=args.post_trace_tau,
            eligibility_tau=args.eligibility_tau,
            reward_prob=args.reward_prob,
            reward_min_len=args.reward_min_len,
            reward_max_len=args.reward_max_len,
            reward_amplitude=args.reward_amplitude,
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
        f"Starting R-STDP stream on ws://{args.host}:{args.port}\n"
        f"  Rates: pre={args.rate_pre}, post_A={args.rate_post_a}, post_B={args.rate_post_b}\n"
        f"  Weight: init={args.w_init}, min={args.w_min}, max={args.w_max}\n"
        f"  Reward windows: prob={args.reward_prob}, "
        f"  len=[{args.reward_min_len}, {args.reward_max_len}], "
        f"  amp={args.reward_amplitude}\n"
        f"  Chunks: {args.chunk_steps} steps, delay={args.delay_ms}ms"
    )

    async with websockets.serve(_handler, args.host, args.port, ping_interval=None):
        await stop.wait()


if __name__ == "__main__":
    asyncio.run(main())
