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
from lava.proc.learning_rules.r_stdp_learning_rule import RewardModulatedSTDP
from rstdp_utils import RSTDPLIF, RSTDPLIFModelFloat, generate_post_spikes


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


@dataclass
class RSimulationState:
    """
    Persistent state for R-STDP network with 1 pre-synaptic and 2 post-synaptic neurons.

    All values are intended to be float and, where possible, normalized
    to the [0, 1] range (membrane threshold, synaptic weights, traces).
    """

    # Plastic weights for connections Pre -> (Post A, Post B), normalized to [w_min, w_max].
    weights: np.ndarray = field(
        default_factory=lambda: np.array([0.2, 0.2], dtype=float)
    )

    # Membrane potentials
    v_pre: float = 0.0
    v_post: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))

    # Currents (u) for LIF neurons
    u_pre: float = 0.0
    u_post: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))

    # Last values of key traces for continuity between chunks
    pre_trace: float = 0.0
    post_trace: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    elig_trace: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    reward_trace: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))

    # Random seed counter for reproducibility across chunks
    seed_counter: int = 0

    # Weight constraints (applied element-wise)
    w_min: float = 0.0
    w_max: float = 1.0

    def clip_weights(self) -> None:
        """Clip all weights to [w_min, w_max] range."""
        self.weights = np.clip(self.weights, self.w_min, self.w_max)

    def to_dict(self) -> Dict:
        """Serialize state to dictionary for debugging / JSON export."""
        return {
            "weights": self.weights.tolist(),
            "v_pre": self.v_pre,
            "v_post": self.v_post.tolist(),
            "u_pre": self.u_pre,
            "u_post": self.u_post.tolist(),
            "pre_trace": self.pre_trace,
            "post_trace": self.post_trace.tolist(),
            "elig_trace": self.elig_trace.tolist(),
            "reward_trace": self.reward_trace.tolist(),
            "seed_counter": self.seed_counter,
            "w_min": self.w_min,
            "w_max": self.w_max,
        }


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
        learning_rate=0.2,        # target max Δw per saturated pre-post event ≈ 0.2
        A_plus=1.0,
        A_minus=-1.0,
        tau_plus=tau_plus,
        tau_minus=tau_minus,
        t_epoch=1,
        x1_impulse=1.0,           # use unit impulse instead of 16
        y1_impulse=1.0,           # use unit impulse instead of 16
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

    # Initialize internal STDP traces (x1: pre, y1: post) from previous state
    plastic.x1.init = np.full(plastic.x1.shape, sim_state.pre_trace)
    plastic.y1.init = np.full(plastic.y1.shape, sim_state.post_trace)

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
    # Readers for real STDP traces inside LearningDense (x1: pre, y1: post)
    x1_reader = Read(buffer=num_steps, interval=1, offset=0)
    y1_reader = Read(buffer=num_steps, interval=1, offset=0)

    v_reader_pre.connect_var(lif_pre.v)
    v_reader_post.connect_var(lif_post.v)
    u_reader_pre.connect_var(lif_pre.u)
    u_reader_post.connect_var(lif_post.u)
    w_reader.connect_var(plastic.weights)
    x1_reader.connect_var(plastic.x1)
    y1_reader.connect_var(plastic.y1)

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
    raw_x1 = np.array(x1_reader.data.get())
    raw_y1 = np.array(y1_reader.data.get())

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
    # Real internal STDP traces from LearningDense (x1: pre-trace, y1: post-trace)
    pre_trace = raw_x1.flatten()
    post_trace = raw_y1.flatten()

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
    pre_trace = pad_array(pre_trace, num_steps)
    post_trace = pad_array(post_trace, num_steps)

    # Apply weight clipping to history (for visualization)
    w_history = np.clip(w_history, sim_state.w_min, sim_state.w_max)

    # ═══════════════════════════════════════════════════════════════════════
    # UPDATE STATE FOR CONTINUITY
    # ═══════════════════════════════════════════════════════════════════════

    sim_state.v_pre = float(v_pre[-1]) if len(v_pre) > 0 else 0.0
    sim_state.v_post = float(v_post[-1]) if len(v_post) > 0 else 0.0
    sim_state.u_pre = float(u_pre[-1]) if len(u_pre) > 0 else 0.0
    sim_state.u_post = float(u_post[-1]) if len(u_post) > 0 else 0.0

    # Get final weight from simulation
    final_w = float(w_history[-1]) if len(w_history) > 0 else sim_state.weight

    # Apply weight decay towards baseline (implemented here because lava-nc
    # doesn't allow pure decay terms without spike dependencies in dw formula)
    # Note: decay is applied to the final weight and ALSO reflected in w_history
    # to avoid jumps between chunks
    if decay_rate > 0:
        alpha = np.exp(-decay_rate * num_steps)
        decayed_w = w_baseline * threshold + (final_w - w_baseline * threshold) * alpha
        sim_state.weight = decayed_w
        # Update last value in history to match decayed weight (smooth transition)
        if len(w_history) > 0:
            w_history[-1] = decayed_w
    else:
        sim_state.weight = final_w

    sim_state.clip_weight()  # Ensure weight is within bounds
    sim_state.seed_counter += 1

    # ═══════════════════════════════════════════════════════════════════════
    # COMPUTE TRACES FOR VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════
    # At this point pre_trace / post_trace already contain the real internal
    # STDP traces (x1, y1) read from LearningDense via x1_reader / y1_reader
    # and padded to num_steps above. For continuity between chunks we only
    # need to remember the last values in SimulationState and feed them back
    # as initial values (see plastic.x1.init / plastic.y1.init).

    sim_state.pre_trace = float(pre_trace[-1]) if len(pre_trace) > 0 else 0.0
    sim_state.post_trace = float(post_trace[-1]) if len(post_trace) > 0 else 0.0

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


def simulate_rstdp(
    num_steps: int = 200,
    threshold: float = 1.0,
    seed: int = 0,
    w_init: float = 0.2,
    w_min: float = 0.0,
    w_max: float = 1.0,
    use_continuous_state: bool = False,
    state: Optional[RSimulationState] = None,
) -> Dict[str, object]:
    """
    Simulate 1 pre-synaptic LIF neuron and 2 post-synaptic RSTDPLIF neurons
    connected by a plastic LearningDense process with RewardModulatedSTDP.

    All signals are floating point and, as far as possible, normalized to
    threshold == 1.0 (membrane potentials, synaptic weights, traces, reward).
    """
    # ═══════════════════════════════════════════════════════════════════════
    # STATE INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════

    if state is not None:
        sim_state = state
    elif use_continuous_state:
        # If global continuity is ever needed, this branch can be extended
        # similarly to the STDP helper. For now we just create a fresh state.
        weights = np.array([w_init, w_init], dtype=float) * threshold
        sim_state = RSimulationState(
            weights=weights,
            w_min=w_min * threshold,
            w_max=w_max * threshold,
        )
    else:
        weights = np.array([w_init, w_init], dtype=float) * threshold
        sim_state = RSimulationState(
            weights=weights,
            w_min=w_min * threshold,
            w_max=w_max * threshold,
        )

    sim_state.clip_weights()

    num_neurons_pre = 1
    num_neurons_post = 2

    du = 1.0
    dv = 1.0

    # ═══════════════════════════════════════════════════════════════════════
    # INPUT GENERATION (normalized)
    # ═══════════════════════════════════════════════════════════════════════

    rng = np.random.default_rng(seed + sim_state.seed_counter)
    spike_prob = np.array([0.03, 0.09], dtype=float)

    spike_raster_pre = np.zeros((num_neurons_pre, num_steps), dtype=int)
    mask = rng.random((num_neurons_pre, num_steps)) < spike_prob[0]
    spike_raster_pre[mask] = 1

    # Generate post-synaptic spikes with pre/post timing structure as in tutorial.
    np.random.seed(seed + sim_state.seed_counter)
    spike_raster_post = generate_post_spikes(spike_raster_pre, num_steps, spike_prob)

    # Graded reward signals for neurons A and B (normalized to [0, 1])
    graded_reward_spikes = np.zeros((num_neurons_post, num_steps), dtype=float)
    graded_reward_spikes[0, 50:70] = 1.0  # Reward A window
    graded_reward_spikes[1, 150:170] = 1.0  # Reward B window

    # ═══════════════════════════════════════════════════════════════════════
    # CONNECTION WEIGHTS (normalized)
    # ═══════════════════════════════════════════════════════════════════════

    wgt_inp_pre = np.eye(num_neurons_pre, dtype=float) * 1.0
    wgt_inp_post = np.eye(num_neurons_post, dtype=float) * 1.0
    wgt_inp_reward = np.eye(num_neurons_post, dtype=float) * 1.0
    wgt_plast_conn = sim_state.weights.reshape(num_neurons_post, num_neurons_pre)

    # ═══════════════════════════════════════════════════════════════════════
    # R-STDP LEARNING RULE
    # ═══════════════════════════════════════════════════════════════════════

    r_stdp = RewardModulatedSTDP(
        learning_rate=0.1,
        A_plus=1.0,
        A_minus=-1.0,
        pre_trace_decay_tau=10.0,
        post_trace_decay_tau=10.0,
        pre_trace_kernel_magnitude=1.0,
        post_trace_kernel_magnitude=1.0,
        eligibility_trace_decay_tau=0.1,
        t_epoch=1,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # PROCESS CREATION
    # ═══════════════════════════════════════════════════════════════════════

    pattern_pre = SpikeIn(data=spike_raster_pre.astype(int))
    pattern_post = SpikeIn(data=spike_raster_post.astype(int))
    reward_pattern_post = SpikeIn(data=graded_reward_spikes.astype(float))

    conn_inp_pre = Dense(weights=wgt_inp_pre)
    conn_inp_post = Dense(weights=wgt_inp_post)
    conn_inp_reward = Dense(weights=wgt_inp_reward, num_message_bits=5)

    lif_pre = LIF(
        shape=(num_neurons_pre,),
        u=sim_state.u_pre,
        v=sim_state.v_pre,
        du=du,
        dv=dv,
        bias_mant=0.0,
        bias_exp=0.0,
        vth=threshold,
        name="lif_pre",
    )

    lif_post = RSTDPLIF(
        shape=(num_neurons_post,),
        u=sim_state.u_post,
        v=sim_state.v_post,
        du=du,
        dv=dv,
        bias_mant=0.0,
        bias_exp=0.0,
        vth=threshold,
        name="lif_post",
        learning_rule=r_stdp,
    )

    plast_conn = LearningDense(
        weights=wgt_plast_conn,
        learning_rule=r_stdp,
        name="plastic_dense",
    )

    # ═══════════════════════════════════════════════════════════════════════
    # MONITORING
    # ═══════════════════════════════════════════════════════════════════════

    v_reader_pre = Read(buffer=num_steps, interval=1, offset=0)
    v_reader_post = Read(buffer=num_steps, interval=1, offset=0)
    u_reader_pre = Read(buffer=num_steps, interval=1, offset=0)
    u_reader_post = Read(buffer=num_steps, interval=1, offset=0)
    w_reader = Read(buffer=num_steps, interval=1, offset=0)
    tag_reader = Read(buffer=num_steps, interval=1, offset=0)
    reward_reader = Read(buffer=num_steps, interval=1, offset=0)
    x1_reader = Read(buffer=num_steps, interval=1, offset=0)
    y1_reader = Read(buffer=num_steps, interval=1, offset=0)

    spike_sink_pre = SinkRing(shape=(num_neurons_pre,), buffer=num_steps)
    spike_sink_post = SinkRing(shape=(num_neurons_post,), buffer=num_steps)

    v_reader_pre.connect_var(lif_pre.v)
    v_reader_post.connect_var(lif_post.v)
    u_reader_pre.connect_var(lif_pre.u)
    u_reader_post.connect_var(lif_post.u)
    w_reader.connect_var(plast_conn.weights)
    tag_reader.connect_var(plast_conn.tag_1)
    reward_reader.connect_var(lif_post.y2)
    x1_reader.connect_var(plast_conn.x1)
    y1_reader.connect_var(lif_post.y1)

    lif_pre.s_out.connect(spike_sink_pre.a_in)
    lif_post.s_out.connect(spike_sink_post.a_in)

    # ═══════════════════════════════════════════════════════════════════════
    # NETWORK TOPOLOGY
    # ═══════════════════════════════════════════════════════════════════════

    pattern_pre.s_out.connect(conn_inp_pre.s_in)
    conn_inp_pre.a_out.connect(lif_pre.a_in)

    pattern_post.s_out.connect(conn_inp_post.s_in)
    conn_inp_post.a_out.connect(lif_post.a_in)

    reward_pattern_post.s_out.connect(conn_inp_reward.s_in)
    conn_inp_reward.a_out.connect(lif_post.a_third_factor_in)

    lif_pre.s_out.connect(plast_conn.s_in)
    plast_conn.a_out.connect(lif_post.a_in)

    lif_post.s_out_bap.connect(plast_conn.s_in_bap)
    lif_post.s_out_y1.connect(plast_conn.s_in_y1)
    lif_post.s_out_y2.connect(plast_conn.s_in_y2)

    # ═══════════════════════════════════════════════════════════════════════
    # RUN SIMULATION
    # ═══════════════════════════════════════════════════════════════════════

    run_cfg = Loihi2SimCfg(select_tag="floating_pt")
    pattern_pre.run(condition=RunSteps(num_steps=num_steps), run_cfg=run_cfg)

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
    raw_tag = np.array(tag_reader.data.get()).astype(float)
    raw_reward = np.array(reward_reader.data.get()).astype(float)
    raw_x1 = np.array(x1_reader.data.get()).astype(float)
    raw_y1 = np.array(y1_reader.data.get()).astype(float)

    pattern_pre.stop()

    # Reshape / flatten to time-series form
    v_pre = raw_v_pre.reshape(1, -1)[0]
    v_post = raw_v_post.reshape(num_neurons_post, -1)
    u_pre = raw_u_pre.reshape(1, -1)[0]
    u_post = raw_u_post.reshape(num_neurons_post, -1)
    s_pre = raw_s_pre.reshape(1, -1)[0]
    s_post = raw_s_post.reshape(num_neurons_post, -1)
    w_hist = raw_w.reshape(num_neurons_post, num_neurons_pre, -1)
    tag_hist = raw_tag.reshape(num_neurons_post, num_neurons_pre, -1)
    reward_hist = raw_reward.reshape(num_neurons_post, -1)
    pre_trace = raw_x1.reshape(-1)
    post_traces = raw_y1.reshape(num_neurons_post, -1)

    # Clip weights history for visualization
    w_hist = np.clip(w_hist, w_min * threshold, w_max * threshold)

    # Take last time-step values for continuity
    sim_state.v_pre = float(v_pre[-1]) if v_pre.size else 0.0
    sim_state.v_post = v_post[:, -1] if v_post.size else np.zeros(2, dtype=float)
    sim_state.u_pre = float(u_pre[-1]) if u_pre.size else 0.0
    sim_state.u_post = u_post[:, -1] if u_post.size else np.zeros(2, dtype=float)

    sim_state.pre_trace = float(pre_trace[-1]) if pre_trace.size else 0.0
    sim_state.post_trace = (
        post_traces[:, -1] if post_traces.size else np.zeros(2, dtype=float)
    )
    sim_state.elig_trace = (
        tag_hist[:, 0, -1] if tag_hist.size else np.zeros(2, dtype=float)
    )
    sim_state.reward_trace = (
        reward_hist[:, -1] if reward_hist.size else np.zeros(2, dtype=float)
    )

    # Final weights for next chunk
    sim_state.weights = w_hist[:, 0, -1] if w_hist.size else sim_state.weights
    sim_state.clip_weights()
    sim_state.seed_counter += 1

    # Prepare neuron-wise data for frontend (three neurons: Pre, Post A, Post B)
    neurons: List[Dict[str, object]] = []
    neurons.append(
        {
            "input_any": spike_raster_pre[0].astype(int).tolist(),
            "membrane_potential": v_pre.tolist(),
            "spikes": s_pre.tolist(),
        }
    )
    neurons.append(
        {
            "input_any": spike_raster_post[0].astype(int).tolist(),
            "membrane_potential": v_post[0].tolist(),
            "spikes": s_post[0].tolist(),
        }
    )
    neurons.append(
        {
            "input_any": spike_raster_post[1].astype(int).tolist(),
            "membrane_potential": v_post[1].tolist(),
            "spikes": s_post[1].tolist(),
        }
    )

    # Prepare R-STDP specific traces for frontend
    rstdp = {
        "pre_trace": pre_trace.tolist(),
        "post_trace_A": post_traces[0].tolist(),
        "post_trace_B": post_traces[1].tolist(),
        "eligibility_A": tag_hist[0, 0].tolist(),
        "eligibility_B": tag_hist[1, 0].tolist(),
        "reward_A": reward_hist[0].tolist(),
        "reward_B": reward_hist[1].tolist(),
        "weight_A": w_hist[0, 0].tolist(),
        "weight_B": w_hist[1, 0].tolist(),
        "w_min": sim_state.w_min,
        "w_max": sim_state.w_max,
    }

    return {
        "dt": 1,
        "threshold": threshold,
        "neurons": neurons,
        "rstdp": rstdp,
        "state": sim_state.to_dict(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate a 3-neuron LIF + R-STDP network (lava-nc) with reward-modulated plasticity."
    )
    parser.add_argument("--steps", type=int, default=200, help="Simulation steps")
    parser.add_argument("--threshold", type=float, default=1.0, help="Spike threshold")

    # Weight management parameters (normalized to threshold)
    parser.add_argument("--w-init", type=float, default=0.2, help="Initial weight as fraction of threshold")
    parser.add_argument("--w-min", type=float, default=0.0, help="Minimum weight as fraction of threshold")
    parser.add_argument("--w-max", type=float, default=1.0, help="Maximum weight as fraction of threshold")

    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--output", type=Path, default=Path("static/data.json"), help="Path to JSON output")
    parser.add_argument("--continuous", action="store_true", help="Use continuous state between runs")
    args = parser.parse_args()

    traces = simulate_rstdp(
        num_steps=args.steps,
        threshold=args.threshold,
        w_init=args.w_init,
        w_min=args.w_min,
        w_max=args.w_max,
        seed=args.seed,
        use_continuous_state=args.continuous,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved R-STDP traces -> {args.output}")
    print(
        f"Final weights: A={traces['rstdp']['weight_A'][-1]:.4f}, "
        f"B={traces['rstdp']['weight_B'][-1]:.4f} "
        f"(bounds: [{traces['rstdp']['w_min']:.2f}, {traces['rstdp']['w_max']:.2f}])"
    )


if __name__ == "__main__":
    main()
