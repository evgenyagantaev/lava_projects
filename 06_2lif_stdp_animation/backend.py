# LIF + STDP network simulation (lava-nc) with JSON export for the animation.
# Run: python backend.py --steps 400 --rate 0.04 --output static/data.json
#
# Architecture (corrected):
#     RND×3 → Dense → LIF_pre → LearningDense → LIF_post ← Dense ← RND×2
#                                    ↑__________________|
#                                           s_in_bap (BAP)

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
from lava.proc.dense.process import Dense, LearningDense
from lava.proc.io.source import RingBuffer as SpikeIn
from lava.proc.io.sink import RingBuffer as SinkRing, Read
from lava.proc.learning_rules.stdp_learning_rule import STDPLoihi
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


def simulate_stdp(
    num_steps: int = 360,
    rate: float = 0.04,
    threshold: float = 1.0,
    spike_fraction: float = 0.4,
    dv: float = 0.04,
    du: float = 1.0,
    bias: float = 0.0,
    seed: int = 0,
    tau_plus: float = 20.0,
    tau_minus: float = 20.0,
) -> Dict[str, object]:
    """
    Simulate 2 LIF neurons with a plastic STDP synapse (neuron0 -> neuron1).

    Corrected architecture with proper BAP (Back-propagating Action Potential)
    connection for STDP learning to work correctly.

    Architecture:
        RND×3 → Dense_pre → LIF_pre → LearningDense → LIF_post ← Dense_post ← RND×2
                                           ↑__________________|
                                                  s_in_bap
    """
    rng = np.random.default_rng(seed)
    spike_amp = threshold * spike_fraction

    # External random spikes: 3 into neuron_pre, 2 into neuron_post
    ext_pre = (rng.random((3, num_steps)) < rate).astype(np.int16)
    ext_post = (rng.random((2, num_steps)) < rate).astype(np.int16)

    # STDP rule (pair-based, floating point)
    stdp = STDPLoihi(
        learning_rate=5.0,
        A_plus=0.05,
        A_minus=0.05,
        tau_plus=tau_plus,
        tau_minus=tau_minus,
        t_epoch=1,
    )

    # Spike input sources
    stim_pre = SpikeIn(data=ext_pre)
    stim_post = SpikeIn(data=ext_post)

    # Static synapses for external inputs
    dense_pre = Dense(weights=np.ones((1, 3)) * spike_amp)
    dense_post = Dense(weights=np.ones((1, 2)) * spike_amp)

    # Two separate LIF neurons (CRITICAL: separate processes for correct BAP)
    lif_pre = LIF(shape=(1,), dv=dv, du=du, vth=threshold, bias_mant=bias)
    lif_post = LIF(shape=(1,), dv=dv, du=du, vth=threshold, bias_mant=bias)

    # Plastic synapse (pre -> post)
    w_init = 0.2 * threshold
    plastic = LearningDense(
        weights=np.array([[w_init]]),
        learning_rule=stdp,
    )

    # Monitoring
    spike_sink_pre = SinkRing(shape=(1,), buffer=num_steps)
    spike_sink_post = SinkRing(shape=(1,), buffer=num_steps)
    v_reader_pre = Read(buffer=num_steps, interval=1, offset=0)
    v_reader_post = Read(buffer=num_steps, interval=1, offset=0)
    w_reader = Read(buffer=num_steps, interval=1, offset=0)

    v_reader_pre.connect_var(lif_pre.v)
    v_reader_post.connect_var(lif_post.v)
    w_reader.connect_var(plastic.weights)

    # ═══════════════════════════════════════════════════════════════════════
    # NETWORK TOPOLOGY
    # ═══════════════════════════════════════════════════════════════════════

    # External inputs -> pre-synaptic neuron
    stim_pre.s_out.connect(dense_pre.s_in)
    dense_pre.a_out.connect(lif_pre.a_in)

    # External inputs -> post-synaptic neuron
    stim_post.s_out.connect(dense_post.s_in)
    dense_post.a_out.connect(lif_post.a_in)

    # Pre-synaptic -> plastic synapse -> post-synaptic
    lif_pre.s_out.connect(plastic.s_in)
    plastic.a_out.connect(lif_post.a_in)

    # ═══════════════════════════════════════════════════════════════════════
    # CRITICAL: Back-propagating Action Potential (BAP) connection
    # Without this, STDP learning DOES NOT WORK!
    # ═══════════════════════════════════════════════════════════════════════
    lif_post.s_out.connect(plastic.s_in_bap)

    # Spike monitoring
    lif_pre.s_out.connect(spike_sink_pre.a_in)
    lif_post.s_out.connect(spike_sink_post.a_in)

    # ═══════════════════════════════════════════════════════════════════════
    # RUN SIMULATION
    # ═══════════════════════════════════════════════════════════════════════

    run_cfg = Loihi2SimCfg(select_tag="floating_pt")
    lif_pre.run(condition=RunSteps(num_steps=num_steps), run_cfg=run_cfg)

    # Collect data
    raw_v_pre = np.array(v_reader_pre.data.get())
    raw_v_post = np.array(v_reader_post.data.get())
    raw_s_pre = np.array(spike_sink_pre.data.get()).astype(int)
    raw_s_post = np.array(spike_sink_post.data.get()).astype(int)
    raw_w = np.array(w_reader.data.get()).astype(float)

    lif_pre.stop()

    # Reshape data
    v_pre = raw_v_pre.flatten()
    v_post = raw_v_post.flatten()
    s_pre = raw_s_pre.flatten()
    s_post = raw_s_post.flatten()
    w_history = raw_w.flatten()

    # Ensure arrays have correct length
    v_pre = v_pre[:num_steps] if len(v_pre) >= num_steps else np.pad(v_pre, (0, num_steps - len(v_pre)))
    v_post = v_post[:num_steps] if len(v_post) >= num_steps else np.pad(v_post, (0, num_steps - len(v_post)))
    s_pre = s_pre[:num_steps] if len(s_pre) >= num_steps else np.pad(s_pre, (0, num_steps - len(s_pre)))
    s_post = s_post[:num_steps] if len(s_post) >= num_steps else np.pad(s_post, (0, num_steps - len(s_post)))
    w_history = w_history[:num_steps] if len(w_history) >= num_steps else np.pad(w_history, (0, num_steps - len(w_history)), constant_values=w_init)

    # Compute traces for UI visualization (synchronized with STDP tau parameters)
    pre_trace = np.zeros(num_steps)
    post_trace = np.zeros(num_steps)
    alpha_pre = np.exp(-1.0 / tau_plus)   # Synchronized with STDP tau_plus
    alpha_post = np.exp(-1.0 / tau_minus)  # Synchronized with STDP tau_minus

    for t in range(num_steps):
        if t > 0:
            pre_trace[t] = pre_trace[t - 1] * alpha_pre
            post_trace[t] = post_trace[t - 1] * alpha_post
        pre_trace[t] += s_pre[t]
        post_trace[t] += s_post[t]

    # Format output for UI compatibility
    neurons: List[Dict[str, object]] = []
    neurons.append(
        {
            "input_any": (ext_pre.sum(axis=0) > 0).astype(int).tolist(),
            "membrane_potential": v_pre.tolist(),
            "spikes": s_pre.tolist(),
        }
    )
    neurons.append(
        {
            # Random spikes on two outer inputs + plastic spike from neuron_pre into middle input
            "input_any": ((ext_post.sum(axis=0) > 0) | (s_pre > 0)).astype(int).tolist(),
            "membrane_potential": v_post.tolist(),
            "spikes": s_post.tolist(),
        }
    )

    return {
        "dt": 1,
        "threshold": threshold,
        "dv": dv,
        "spike_amplitude": float(spike_amp),
        "neurons": neurons,
        "inputs_detail": [
            ext_pre.astype(int).tolist(),  # neuron_pre: 3 random inputs
            np.stack([ext_post[0], s_pre, ext_post[1]]).astype(int).tolist(),  # neuron_post: left rand, middle plastic, right rand
        ],
        "stdp": {
            "pre_trace": pre_trace.tolist(),
            "post_trace": post_trace.tolist(),
            "weight": w_history.tolist(),
            "tau_plus": tau_plus,
            "tau_minus": tau_minus,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate a 2-neuron LIF + STDP network (lava-nc) and export traces for the animation UI."
    )
    parser.add_argument("--steps", type=int, default=360, help="Simulation steps")
    parser.add_argument("--rate", type=float, default=0.04, help="Probability of external spike per tick")
    parser.add_argument("--threshold", type=float, default=1.0, help="Spike threshold")
    parser.add_argument("--spike-fraction", type=float, default=0.4, help="External spike amplitude as fraction of threshold")
    parser.add_argument("--dv", type=float, default=0.04, help="Leak factor for membrane potential")
    parser.add_argument("--bias", type=float, default=0.0, help="Constant bias for membrane potential")
    parser.add_argument("--tau-plus", type=float, default=20.0, help="STDP tau_plus (pre-synaptic trace decay)")
    parser.add_argument("--tau-minus", type=float, default=20.0, help="STDP tau_minus (post-synaptic trace decay)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--output", type=Path, default=Path("static/data.json"), help="Path to JSON output")
    args = parser.parse_args()

    traces = simulate_stdp(
        num_steps=args.steps,
        rate=args.rate,
        threshold=args.threshold,
        spike_fraction=args.spike_fraction,
        dv=args.dv,
        bias=args.bias,
        tau_plus=args.tau_plus,
        tau_minus=args.tau_minus,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved LIF traces -> {args.output}")


if __name__ == "__main__":
    main()
