"""Realtime LIF stream server backed by lava-nc (two independent neurons).

Runs a WebSocket server that streams traces of two floating-point LIF neurons
simulated via lava-nc (Loihi2Sim floating_pt). Frames are generated in chunks
and emitted in realtime.

Run:
  python projects/05_2lif_animation/server.py --host 127.0.0.1 --port 8765
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

from backend import simulate_pair


async def lif_stream(rate: float, dv: float, threshold: float, spike_amp: float, delay_ms: int, chunk_steps: int) -> AsyncIterator[Dict[str, float]]:
    """Generate frames for two LIF neurons in chunks without blocking the loop."""
    t = 0
    seed = 0

    while True:
        traces = await asyncio.to_thread(
            simulate_pair,
            num_steps=chunk_steps,
            rate=rate,
            threshold=threshold,
            spike_fraction=spike_amp / threshold,
            dv=dv,
            seed=seed,
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
            }
            t += 1
            await asyncio.sleep(delay_ms / 1000.0)


async def handler(websocket, *, rate: float, dv: float, threshold: float, spike_amp: float, delay_ms: int, chunk_steps: int):
    async for frame in lif_stream(rate=rate, dv=dv, threshold=threshold, spike_amp=spike_amp, delay_ms=delay_ms, chunk_steps=chunk_steps):
        await websocket.send(json.dumps(frame))


async def main():
    parser = argparse.ArgumentParser(description="Realtime LIF WebSocket streamer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--rate", type=float, default=0.05, help="Вероятность входного спайка на такт (уменьшено в 2 раза)")
    parser.add_argument("--dv", type=float, default=0.04, help="Обратная постоянная утечки")
    parser.add_argument("--threshold", type=float, default=1.0, help="Порог срабатывания")
    parser.add_argument("--spike-fraction", type=float, default=0.4, help="Амплитуда входного спайка как доля порога")
    parser.add_argument("--delay-ms", type=int, default=80, help="Задержка между шагами для визуализации")
    parser.add_argument("--chunk-steps", type=int, default=256, help="Сколько шагов симулировать за раз (меньше — быстрее первые кадры)")
    args = parser.parse_args()

    spike_amp = args.threshold * args.spike_fraction

    async def _handler(ws):
        return await handler(ws, rate=args.rate, dv=args.dv, threshold=args.threshold, spike_amp=spike_amp, delay_ms=args.delay_ms, chunk_steps=args.chunk_steps)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    # Unix-friendly graceful shutdown
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
    except NotImplementedError:
        # Windows ProactorEventLoop does not support add_signal_handler; fallback to signal.signal
        def _handle(_sig, _frame):
            stop.set()
        signal.signal(signal.SIGINT, _handle)

    print(
        f"Starting LIF stream on ws://{args.host}:{args.port}/stream "
        f"(dv={args.dv}, spike_amp={spike_amp}, chunk_steps={args.chunk_steps}, delay_ms={args.delay_ms})"
    )
    async with websockets.serve(_handler, args.host, args.port, ping_interval=None):
        await stop.wait()


if __name__ == "__main__":
    asyncio.run(main())
