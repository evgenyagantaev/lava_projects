# LIF neuron simulation with lava-nc plus JSON export for the HTML animation.
# Run: python backend.py --steps 400 --rate 0.04 --output static/data.json

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

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


def simulate_pair(num_steps: int = 360, rate: float = 0.04, threshold: float = 1.0,
                  spike_fraction: float = 0.4, dv: float = 0.04, du: float = 1.0,
                  bias: float = 0.0, seed: int = 0, n_inputs: int = 6) -> Dict[str, object]:
    """Simulate one 2-neuron network with 6 inputs (3 per neuron) and return traces."""
    spike_amp = threshold * spike_fraction
    rng = np.random.default_rng(seed)

    # 6 random inputs for the whole network; first 3 -> neuron 0, next 3 -> neuron 1
    input_spikes = (rng.random((n_inputs, num_steps)) < rate).astype(np.int16)

    weights = np.zeros((2, n_inputs), dtype=float)
    weights[0, :3] = spike_amp
    weights[1, 3:6] = spike_amp

    stimulus = SpikeIn(data=input_spikes)
    syn = Dense(weights=weights)
    lif = LIF(shape=(2,), dv=dv, du=du, vth=threshold, bias_mant=bias)
    spike_sink = SinkRing(shape=(2,), buffer=num_steps)
    v_reader = Read(buffer=num_steps, interval=1, offset=0)
    v_reader.connect_var(lif.v)

    stimulus.s_out.connect(syn.s_in)
    syn.a_out.connect(lif.a_in)
    lif.s_out.connect(spike_sink.a_in)

    run_cfg = Loihi2SimCfg(select_tag="floating_pt")
    lif.run(condition=RunSteps(num_steps=num_steps), run_cfg=run_cfg)
    raw_v = np.array(v_reader.data.get())
    raw_s = np.array(spike_sink.data.get()).astype(int)
    v_data = raw_v.reshape(num_steps, 2)
    s_data = raw_s.reshape(num_steps, 2)
    lif.stop()

    neurons: List[Dict[str, object]] = []
    for nid in range(2):
        start = nid * 3
        end = start + 3
        neurons.append({
            "input_any": (input_spikes[start:end].sum(axis=0) > 0).astype(int).tolist(),
            "membrane_potential": v_data[:, nid].tolist(),
            "spikes": s_data[:, nid].tolist(),
        })

    return {
        "dt": 1,
        "threshold": threshold,
        "dv": dv,
        "spike_amplitude": float(spike_amp),
        "neurons": neurons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a LIF neuron with lava-nc and export traces for the animation UI.")
    parser.add_argument("--steps", type=int, default=360, help="Количество шагов симуляции")
    parser.add_argument("--rate", type=float, default=0.04, help="Вероятность входного спайка на такт (понижена х2)")
    parser.add_argument("--threshold", type=float, default=1.0, help="Порог срабатывания нейрона")
    parser.add_argument("--spike-fraction", type=float, default=0.4, help="Амплитуда входного спайка как доля порога")
    parser.add_argument("--dv", type=float, default=0.04, help="Обратная постоянная утечки мембранного потенциала")
    parser.add_argument("--bias", type=float, default=0.0, help="Тонкий сдвиг мембранного потенциала")
    parser.add_argument("--seed", type=int, default=0, help="Инициализация генератора случайных чисел")
    parser.add_argument("--output", type=Path, default=Path("static/data.json"), help="Путь к JSON с результатами")
    args = parser.parse_args()

    traces = simulate_pair(num_steps=args.steps, rate=args.rate, threshold=args.threshold,
                           spike_fraction=args.spike_fraction, dv=args.dv, bias=args.bias,
                           seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved LIF traces -> {args.output}")


if __name__ == "__main__":
    main()
