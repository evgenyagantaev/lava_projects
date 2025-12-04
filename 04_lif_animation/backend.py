# LIF neuron simulation with lava-nc plus JSON export for the HTML animation.
# Run: python backend.py --steps 400 --rate 0.08 --output static/data.json

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

# Prefer installed lava-nc over local source tree to avoid picking up repo sources accidentally
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    p_str = str(p)
    if p_str in sys.path:
        sys.path.remove(p_str)

from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.proc.io.source import RingBuffer as SpikeIn
from lava.proc.io.sink import RingBuffer as SinkRing, Read
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


class SpikeSink(AbstractProcess):
    """Simple sink to capture latest spikes without Monitor."""

    def __init__(self, *, shape=(1,)):
        super().__init__()
        self.s_in = InPort(shape=shape)
        self.last = Var(shape=shape, init=0)


@implements(proc=SpikeSink, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PySpikeSinkModel(PyLoihiProcessModel):
    s_in = LavaPyType(PyInPort.VEC_DENSE, np.int32)
    last: np.ndarray = LavaPyType(np.ndarray, np.int32)

    def run_spk(self) -> None:
        data = self.s_in.recv()
        self.last.assign(data)


def simulate_lif(num_steps: int = 360, rate: float = 0.08, threshold: float = 1.0,
                 spike_fraction: float = 0.4, dv: float = 0.04, bias: float = 0.0,
                 seed: int = 0) -> Dict[str, object]:
    # Simulate a single floating-point LIF neuron and return traces as lists.
    rng = np.random.default_rng(seed)
    input_spikes = (rng.random(num_steps) < rate).astype(np.int16).reshape(1, num_steps)
    spike_amp = threshold * spike_fraction

    stimulus = SpikeIn(data=input_spikes)
    syn = Dense(weights=np.array([[spike_amp]], dtype=float))
    lif = LIF(shape=(1,), dv=dv, du=0.0, vth=threshold, bias_mant=bias)
    spike_sink = SinkRing(shape=(1,), buffer=num_steps)
    v_reader = Read(buffer=num_steps, interval=1, offset=0)
    v_reader.connect_var(lif.v)

    stimulus.s_out.connect(syn.s_in)
    syn.a_out.connect(lif.a_in)
    lif.s_out.connect(spike_sink.a_in)

    run_cfg = Loihi2SimCfg(select_tag="floating_pt")
    lif.run(condition=RunSteps(num_steps=num_steps), run_cfg=run_cfg)
    v_trace = v_reader.data.get().flatten().tolist()
    s_trace = spike_sink.data.get().astype(int).flatten().tolist()
    lif.stop()

    return {
        "dt": 1,
        "threshold": threshold,
        "dv": dv,
        "spike_amplitude": float(spike_amp),
        "input_spikes": input_spikes.flatten().astype(int).tolist(),
        "membrane_potential": v_trace,
        "spikes": s_trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a LIF neuron with lava-nc and export traces for the animation UI.")
    parser.add_argument("--steps", type=int, default=360, help="Количество шагов симуляции")
    parser.add_argument("--rate", type=float, default=0.08, help="Вероятность входного спайка на такт")
    parser.add_argument("--threshold", type=float, default=1.0, help="Порог срабатывания нейрона")
    parser.add_argument("--spike-fraction", type=float, default=0.4, help="Амплитуда входного спайка как доля порога")
    parser.add_argument("--dv", type=float, default=0.04, help="Обратная постоянная утечки мембранного потенциала")
    parser.add_argument("--bias", type=float, default=0.0, help="Тонкий сдвиг мембранного потенциала")
    parser.add_argument("--seed", type=int, default=0, help="Инициализация генератора случайных чисел")
    parser.add_argument("--output", type=Path, default=Path("static/data.json"), help="Путь к JSON с результатами")
    args = parser.parse_args()

    traces = simulate_lif(num_steps=args.steps, rate=args.rate, threshold=args.threshold,
                          spike_fraction=args.spike_fraction, dv=args.dv, bias=args.bias,
                          seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved LIF traces -> {args.output}")


if __name__ == "__main__":
    main()
