# LIF + STDP network simulation (lava-nc) with JSON export for the animation.
# Run: python backend.py --steps 400 --rate 0.04 --output static/data.json
#
# Architecture:
#     RND×3 → Dense → LIF_pre → LearningDense → LIF_post ← Dense ← RND×2
#                                    ↑__________________|
#                                           s_in_bap (BAP)
#
# Features:
#   - Weight clipping (w_min, w_max)
#   - Weight decay towards baseline
#   - State persistence between simulation chunks

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Prefer installed lava-nc over local source tree
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


# ═══════════════════════════════════════════════════════════════════════════════
# STDP LEARNING RULE (using standard STDPLoihi)
# ═══════════════════════════════════════════════════════════════════════════════
# Note: Weight decay is implemented via post-processing between simulation chunks,
# not in the learning rule formula, because lava-nc requires each term in dw to
# have a dependency (x0, y0, or u). Pure decay terms like "-decay * w" are not allowed.

from lava.proc.learning_rules.stdp_learning_rule import STDPLoihi


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION STATE FOR CONTINUITY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationState:
    """
    Persistent state for continuous simulation across multiple chunks.

    This allows the simulation to maintain continuity of:
    - Synaptic weights
    - Membrane potentials
    - Trace values (approximated for visualization)
    """
    # Synaptic weight (scalar for 1->1 connection)
    weight: float = 0.2

    # Membrane potentials
    v_pre: float = 0.0
    v_post: float = 0.0

    # Current (u) for LIF neurons
    u_pre: float = 0.0
    u_post: float = 0.0

    # Traces for visualization (approximation)
    pre_trace: float = 0.0
    post_trace: float = 0.0

    # Random seed counter for reproducibility
    seed_counter: int = 0

    # Weight constraints
    w_min: float = 0.0
    w_max: float = 1.0

    def clip_weight(self) -> None:
        """Clip weight to [w_min, w_max] range."""
        self.weight = np.clip(self.weight, self.w_min, self.w_max)

    def apply_decay(self, decay_rate: float, w_baseline: float, num_steps: int) -> None:
        """
        Apply weight decay towards baseline.

        Formula: w_new = w + decay_rate * num_steps * (w_baseline - w)
        This is equivalent to exponential decay: w → w_baseline as t → ∞

        Parameters
        ----------
        decay_rate : float
            Decay rate per time step (e.g., 0.001).
        w_baseline : float
            Target baseline weight.
        num_steps : int
            Number of simulation steps in this chunk.
        """
        # Exponential decay towards baseline
        alpha = np.exp(-decay_rate * num_steps)
        self.weight = w_baseline + (self.weight - w_baseline) * alpha

    def to_dict(self) -> Dict:
        """Serialize state to dictionary."""
        return {
            "weight": self.weight,
            "v_pre": self.v_pre,
            "v_post": self.v_post,
            "u_pre": self.u_pre,
            "u_post": self.u_post,
            "pre_trace": self.pre_trace,
            "post_trace": self.post_trace,
            "seed_counter": self.seed_counter,
            "w_min": self.w_min,
            "w_max": self.w_max,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SimulationState":
        """Deserialize state from dictionary."""
        return cls(**data)


# Global state for server mode
_global_state: Optional[SimulationState] = None


def get_or_create_state(
    w_init: float = 0.2,
    w_min: float = 0.0,
    w_max: float = 1.0,
) -> SimulationState:
    """Get existing state or create new one."""
    global _global_state
    if _global_state is None:
        _global_state = SimulationState(
            weight=w_init,
            w_min=w_min,
            w_max=w_max,
        )
    return _global_state


def reset_state() -> None:
    """Reset global state to None."""
    global _global_state
    _global_state = None


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM PROCESSES
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

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
    # New parameters for weight management
    w_init: float = 0.2,
    w_min: float = 0.0,
    w_max: float = 1.0,
    decay_rate: float = 0.001,
    w_baseline: float = 0.1,
    # State management
    use_continuous_state: bool = False,
    state: Optional[SimulationState] = None,
) -> Dict[str, object]:
    """
    Simulate 2 LIF neurons with a plastic STDP synapse (neuron0 -> neuron1).

    Features:
    - Weight clipping: weights are bounded to [w_min, w_max]
    - Weight decay: weights slowly drift towards w_baseline
    - State continuity: membrane potentials and weights persist between chunks

    Parameters
    ----------
    num_steps : int
        Number of simulation steps.
    w_min : float
        Minimum allowed weight value (default: 0.0).
    w_max : float
        Maximum allowed weight value (default: 1.0 = threshold).
    decay_rate : float
        Rate of weight decay towards baseline (default: 0.001).
    w_baseline : float
        Baseline weight value for decay (default: 0.1).
    use_continuous_state : bool
        If True, use global state for continuity between calls.
    state : SimulationState, optional
        External state object (overrides global state if provided).
    """
    # ═══════════════════════════════════════════════════════════════════════
    # STATE INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════

    if state is not None:
        sim_state = state
    elif use_continuous_state:
        sim_state = get_or_create_state(w_init * threshold, w_min * threshold, w_max * threshold)
    else:
        sim_state = SimulationState(
            weight=w_init * threshold,
            w_min=w_min * threshold,
            w_max=w_max * threshold,
        )

    # Apply weight clipping from previous run
    sim_state.clip_weight()

    rng = np.random.default_rng(seed + sim_state.seed_counter)
    spike_amp = threshold * spike_fraction

    # ═══════════════════════════════════════════════════════════════════════
    # INPUT GENERATION
    # ═══════════════════════════════════════════════════════════════════════

    ext_pre = (rng.random((3, num_steps)) < rate).astype(np.int16)
    ext_post = (rng.random((2, num_steps)) < rate).astype(np.int16)

    # ═══════════════════════════════════════════════════════════════════════
    # STDP LEARNING RULE
    # ═══════════════════════════════════════════════════════════════════════
    # Note: Weight decay is applied separately via SimulationState.apply_decay()

    stdp = STDPLoihi(
        learning_rate=5.0,
        A_plus=0.05,
        A_minus=0.05,
        tau_plus=tau_plus,
        tau_minus=tau_minus,
        t_epoch=1,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # PROCESS CREATION
    # ═══════════════════════════════════════════════════════════════════════

    stim_pre = SpikeIn(data=ext_pre)
    stim_post = SpikeIn(data=ext_post)

    dense_pre = Dense(weights=np.ones((1, 3)) * spike_amp)
    dense_post = Dense(weights=np.ones((1, 2)) * spike_amp)

    # Initialize LIF with previous state
    lif_pre = LIF(
        shape=(1,),
        u=sim_state.u_pre,
        v=sim_state.v_pre,
        dv=dv,
        du=du,
        vth=threshold,
        bias_mant=bias,
    )
    lif_post = LIF(
        shape=(1,),
        u=sim_state.u_post,
        v=sim_state.v_post,
        dv=dv,
        du=du,
        vth=threshold,
        bias_mant=bias,
    )

    # Initialize plastic synapse with previous weight
    plastic = LearningDense(
        weights=np.array([[sim_state.weight]]),
        learning_rule=stdp,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # MONITORING
    # ═══════════════════════════════════════════════════════════════════════

    spike_sink_pre = SinkRing(shape=(1,), buffer=num_steps)
    spike_sink_post = SinkRing(shape=(1,), buffer=num_steps)
    v_reader_pre = Read(buffer=num_steps, interval=1, offset=0)
    v_reader_post = Read(buffer=num_steps, interval=1, offset=0)
    u_reader_pre = Read(buffer=num_steps, interval=1, offset=0)
    u_reader_post = Read(buffer=num_steps, interval=1, offset=0)
    w_reader = Read(buffer=num_steps, interval=1, offset=0)

    v_reader_pre.connect_var(lif_pre.v)
    v_reader_post.connect_var(lif_post.v)
    u_reader_pre.connect_var(lif_pre.u)
    u_reader_post.connect_var(lif_post.u)
    w_reader.connect_var(plastic.weights)

    # ═══════════════════════════════════════════════════════════════════════
    # NETWORK TOPOLOGY
    # ═══════════════════════════════════════════════════════════════════════

    stim_pre.s_out.connect(dense_pre.s_in)
    dense_pre.a_out.connect(lif_pre.a_in)

    stim_post.s_out.connect(dense_post.s_in)
    dense_post.a_out.connect(lif_post.a_in)

    lif_pre.s_out.connect(plastic.s_in)
    plastic.a_out.connect(lif_post.a_in)

    # CRITICAL: BAP connection for STDP
    lif_post.s_out.connect(plastic.s_in_bap)

    lif_pre.s_out.connect(spike_sink_pre.a_in)
    lif_post.s_out.connect(spike_sink_post.a_in)

    # ═══════════════════════════════════════════════════════════════════════
    # RUN SIMULATION
    # ═══════════════════════════════════════════════════════════════════════

    run_cfg = Loihi2SimCfg(select_tag="floating_pt")
    lif_pre.run(condition=RunSteps(num_steps=num_steps), run_cfg=run_cfg)

    # ═══════════════════════════════════════════════════════════════════════
    # COLLECT DATA
    # ═══════════════════════════════════════════════════════════════════════

    raw_v_pre = np.array(v_reader_pre.data.get())
    raw_v_post = np.array(v_reader_post.data.get())
    raw_u_pre = np.array(u_reader_pre.data.get())
    raw_u_post = np.array(u_reader_post.data.get())
    raw_s_pre = np.array(spike_sink_pre.data.get()).astype(int)
    raw_s_post = np.array(spike_sink_post.data.get()).astype(int)
    raw_w = np.array(w_reader.data.get()).astype(float)

    lif_pre.stop()

    # ═══════════════════════════════════════════════════════════════════════
    # PROCESS AND CLIP DATA
    # ═══════════════════════════════════════════════════════════════════════

    v_pre = raw_v_pre.flatten()
    v_post = raw_v_post.flatten()
    u_pre = raw_u_pre.flatten()
    u_post = raw_u_post.flatten()
    s_pre = raw_s_pre.flatten()
    s_post = raw_s_post.flatten()
    w_history = raw_w.flatten()

    # Ensure arrays have correct length
    def pad_array(arr, length, fill_value=0):
        if len(arr) >= length:
            return arr[:length]
        return np.pad(arr, (0, length - len(arr)), constant_values=fill_value)

    v_pre = pad_array(v_pre, num_steps)
    v_post = pad_array(v_post, num_steps)
    u_pre = pad_array(u_pre, num_steps)
    u_post = pad_array(u_post, num_steps)
    s_pre = pad_array(s_pre, num_steps)
    s_post = pad_array(s_post, num_steps)
    w_history = pad_array(w_history, num_steps, sim_state.weight)

    # Apply weight clipping to history (for visualization)
    w_history = np.clip(w_history, sim_state.w_min, sim_state.w_max)

    # ═══════════════════════════════════════════════════════════════════════
    # UPDATE STATE FOR CONTINUITY
    # ═══════════════════════════════════════════════════════════════════════

    sim_state.v_pre = float(v_pre[-1]) if len(v_pre) > 0 else 0.0
    sim_state.v_post = float(v_post[-1]) if len(v_post) > 0 else 0.0
    sim_state.u_pre = float(u_pre[-1]) if len(u_pre) > 0 else 0.0
    sim_state.u_post = float(u_post[-1]) if len(u_post) > 0 else 0.0
    sim_state.weight = float(w_history[-1]) if len(w_history) > 0 else sim_state.weight

    # Apply weight decay towards baseline (implemented here because lava-nc
    # doesn't allow pure decay terms without spike dependencies in dw formula)
    if decay_rate > 0:
        sim_state.apply_decay(decay_rate, w_baseline * threshold, num_steps)

    sim_state.clip_weight()  # Ensure weight is within bounds
    sim_state.seed_counter += 1

    # ═══════════════════════════════════════════════════════════════════════
    # COMPUTE TRACES FOR VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════

    pre_trace = np.zeros(num_steps)
    post_trace = np.zeros(num_steps)
    alpha_pre = np.exp(-1.0 / tau_plus)
    alpha_post = np.exp(-1.0 / tau_minus)

    # Start with previous trace values
    current_pre_trace = sim_state.pre_trace
    current_post_trace = sim_state.post_trace

    for t in range(num_steps):
        current_pre_trace = current_pre_trace * alpha_pre + s_pre[t]
        current_post_trace = current_post_trace * alpha_post + s_post[t]
        pre_trace[t] = current_pre_trace
        post_trace[t] = current_post_trace

    # Save final trace values for continuity
    sim_state.pre_trace = current_pre_trace
    sim_state.post_trace = current_post_trace

    # ═══════════════════════════════════════════════════════════════════════
    # FORMAT OUTPUT
    # ═══════════════════════════════════════════════════════════════════════

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
            ext_pre.astype(int).tolist(),
            np.stack([ext_post[0], s_pre, ext_post[1]]).astype(int).tolist(),
        ],
        "stdp": {
            "pre_trace": pre_trace.tolist(),
            "post_trace": post_trace.tolist(),
            "weight": w_history.tolist(),
            "tau_plus": tau_plus,
            "tau_minus": tau_minus,
            "w_min": sim_state.w_min,
            "w_max": sim_state.w_max,
            "decay_rate": decay_rate,
            "w_baseline": w_baseline * threshold,
        },
        # Include state for debugging/inspection
        "state": sim_state.to_dict(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate a 2-neuron LIF + STDP network (lava-nc) with weight management."
    )
    parser.add_argument("--steps", type=int, default=360, help="Simulation steps")
    parser.add_argument("--rate", type=float, default=0.04, help="Probability of external spike per tick")
    parser.add_argument("--threshold", type=float, default=1.0, help="Spike threshold")
    parser.add_argument("--spike-fraction", type=float, default=0.4, help="External spike amplitude as fraction of threshold")
    parser.add_argument("--dv", type=float, default=0.04, help="Leak factor for membrane potential")
    parser.add_argument("--bias", type=float, default=0.0, help="Constant bias for membrane potential")
    parser.add_argument("--tau-plus", type=float, default=20.0, help="STDP tau_plus (pre-synaptic trace decay)")
    parser.add_argument("--tau-minus", type=float, default=20.0, help="STDP tau_minus (post-synaptic trace decay)")

    # Weight management parameters
    parser.add_argument("--w-init", type=float, default=0.2, help="Initial weight as fraction of threshold")
    parser.add_argument("--w-min", type=float, default=0.0, help="Minimum weight as fraction of threshold")
    parser.add_argument("--w-max", type=float, default=1.0, help="Maximum weight as fraction of threshold")
    parser.add_argument("--decay-rate", type=float, default=0.001, help="Weight decay rate towards baseline")
    parser.add_argument("--w-baseline", type=float, default=0.1, help="Baseline weight for decay (fraction of threshold)")

    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--output", type=Path, default=Path("static/data.json"), help="Path to JSON output")
    parser.add_argument("--continuous", action="store_true", help="Use continuous state between runs")
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
        w_init=args.w_init,
        w_min=args.w_min,
        w_max=args.w_max,
        decay_rate=args.decay_rate,
        w_baseline=args.w_baseline,
        seed=args.seed,
        use_continuous_state=args.continuous,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved LIF traces -> {args.output}")
    print(f"Final weight: {traces['state']['weight']:.4f} (bounds: [{traces['stdp']['w_min']:.2f}, {traces['stdp']['w_max']:.2f}])")


if __name__ == "__main__":
    main()
