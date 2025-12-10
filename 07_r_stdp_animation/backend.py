# R-STDP network simulation (lava-nc) with JSON export for the animation.
# Run: python backend.py --steps 200 --output static/data.json
#
# Architecture (based on tutorial01_Reward_Modulated_STDP.ipynb):
#
#     RND → Dense → LIF_pre ─┬─ LearningDense(w1) ──► RSTDPLIF_A ◄── Dense ← RND
#                            │                              ↑
#                            │                        s_in_bap, s_in_y1, s_in_y2
#                            │
#                            └─ LearningDense(w2) ──► RSTDPLIF_B ◄── Dense ← RND
#                                                           ↑
#                                                     s_in_bap, s_in_y1, s_in_y2
#
#     Graded Reward A ────► Dense(num_message_bits) ──► RSTDPLIF_A.a_third_factor_in
#     Graded Reward B ────► Dense(num_message_bits) ──► RSTDPLIF_B.a_third_factor_in
#
# Features:
#   - Weight clipping (w_min, w_max)
#   - Eligibility traces (tag_1) for each synapse
#   - Localized graded reward signals
#   - State persistence between simulation chunks
#   - All values normalized to ~[0, 1] range

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Prefer installed lava-nc over local source tree
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    p_str = str(p)
    if p_str in sys.path:
        sys.path.remove(p_str)

from lava.proc.lif.process import LIF, LearningLIF
from lava.proc.dense.process import Dense, LearningDense
from lava.proc.io.source import RingBuffer as SpikeIn
from lava.proc.io.sink import RingBuffer as SinkRing, Read
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.neuron import LearningNeuronModelFloat
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.lif.models import AbstractPyLifModelFloat

# R-STDP learning rule from Lava
from lava.proc.learning_rules.r_stdp_learning_rule import RewardModulatedSTDP


# ═══════════════════════════════════════════════════════════════════════════════
# RSTDPLIF PROCESS (from tutorials/in_depth/three_factor_learning/utils.py)
# ═══════════════════════════════════════════════════════════════════════════════
# This is a LIF neuron that computes post-synaptic traces and reward traces
# needed for R-STDP learning.

class RSTDPLIF(LearningLIF):
    """LIF neuron with R-STDP support (third-factor learning)."""
    pass


@implements(proc=RSTDPLIF, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class RSTDPLIFModelFloat(LearningNeuronModelFloat, AbstractPyLifModelFloat):
    """
    Floating-point implementation of RSTDPLIF for R-STDP.

    Computes:
    - y1: post-synaptic trace (based on output spikes)
    - y2: reward trace (third factor, from graded input)

    Sends traces to LearningDense via s_out_y1 and s_out_y2 ports.
    """

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    vth: float = LavaPyType(float, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.s_out_buff = np.zeros(proc_params["shape"])

    def spiking_activation(self):
        """Spike when voltage exceeds threshold."""
        return self.v > self.vth

    def calculate_third_factor_trace(self, s_graded_in: float) -> float:
        """
        Generate third factor (reward) trace from graded input spikes.
        The reward trace simply mirrors the graded input.
        """
        return s_graded_in

    def compute_post_synaptic_trace(self, s_out_buff):
        """
        Compute post-synaptic trace (y1) for this time step.

        y1_new = y1_old * exp(-1/tau) + impulse * spike
        """
        y1_tau = self._learning_rule.post_trace_decay_tau
        y1_impulse = self._learning_rule.post_trace_kernel_magnitude

        return self.y1 * np.exp(-1 / y1_tau) + y1_impulse * s_out_buff

    def run_spk(self) -> None:
        """
        Run spiking phase:
        1. Compute post-synaptic trace y1
        2. Call parent run_spk (LIF dynamics)
        3. Receive graded reward input and compute y2
        4. Send BAP, y1, y2, y3 to LearningDense
        """
        # Compute y1 trace before parent run_spk
        self.y1 = self.compute_post_synaptic_trace(self.s_out_buff)

        # Standard LIF dynamics
        super().run_spk()

        # Receive graded reward input
        a_graded_in = self.a_third_factor_in.recv()

        # Compute y2 (reward trace)
        self.y2 = self.calculate_third_factor_trace(a_graded_in)

        # Send traces to LearningDense
        self.s_out_bap.send(self.s_out_buff)
        self.s_out_y1.send(self.y1)
        self.s_out_y2.send(self.y2)
        self.s_out_y3.send(self.y3)


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION STATE FOR CONTINUITY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationState:
    """
    Persistent state for continuous R-STDP simulation across multiple chunks.

    Stores:
    - Two synaptic weights (w_A, w_B) for connections to post neurons A and B
    - Membrane potentials and currents for all three neurons
    - Pre-synaptic trace (x1)
    - Post-synaptic traces (y1) for neurons A and B
    - Eligibility traces for both synapses
    - Reward traces for neurons A and B
    """
    # Synaptic weights (normalized to [0, 1])
    w_A: float = 0.5
    w_B: float = 0.5

    # Membrane potentials
    v_pre: float = 0.0
    v_post_A: float = 0.0
    v_post_B: float = 0.0

    # Currents (u) for LIF neurons
    u_pre: float = 0.0
    u_post_A: float = 0.0
    u_post_B: float = 0.0

    # Pre-synaptic trace (x1)
    pre_trace: float = 0.0

    # Post-synaptic traces (y1) for neurons A and B
    post_trace_A: float = 0.0
    post_trace_B: float = 0.0

    # Eligibility traces (tag_1) for synapses to A and B
    eligibility_A: float = 0.0
    eligibility_B: float = 0.0

    # Reward traces for neurons A and B
    reward_A: float = 0.0
    reward_B: float = 0.0

    # Random seed counter for reproducibility
    seed_counter: int = 0

    # Weight constraints
    w_min: float = 0.0
    w_max: float = 1.0

    def clip_weights(self) -> None:
        """Clip both weights to [w_min, w_max] range."""
        self.w_A = np.clip(self.w_A, self.w_min, self.w_max)
        self.w_B = np.clip(self.w_B, self.w_min, self.w_max)

    def to_dict(self) -> Dict:
        """Serialize state to dictionary."""
        return {
            "w_A": self.w_A,
            "w_B": self.w_B,
            "v_pre": self.v_pre,
            "v_post_A": self.v_post_A,
            "v_post_B": self.v_post_B,
            "u_pre": self.u_pre,
            "u_post_A": self.u_post_A,
            "u_post_B": self.u_post_B,
            "pre_trace": self.pre_trace,
            "post_trace_A": self.post_trace_A,
            "post_trace_B": self.post_trace_B,
            "eligibility_A": self.eligibility_A,
            "eligibility_B": self.eligibility_B,
            "reward_A": self.reward_A,
            "reward_B": self.reward_B,
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
    w_init: float = 0.5,
    w_min: float = 0.0,
    w_max: float = 1.0,
) -> SimulationState:
    """Get existing state or create new one."""
    global _global_state
    if _global_state is None:
        _global_state = SimulationState(
            w_A=w_init,
            w_B=w_init,
            w_min=w_min,
            w_max=w_max,
        )
    return _global_state


def reset_state() -> None:
    """Reset global state to None."""
    global _global_state
    _global_state = None


# ═══════════════════════════════════════════════════════════════════════════════
# SPIKE GENERATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def generate_post_spikes(
    pre_spike_times: np.ndarray,
    num_steps: int,
    spike_prob: List[float],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate post-synaptic spikes correlated with pre-synaptic spikes.

    For neuron A: spikes tend to occur AFTER pre-synaptic spikes (potentiation)
    For neuron B: spikes tend to occur BEFORE pre-synaptic spikes (depression)

    Parameters
    ----------
    pre_spike_times : np.ndarray
        Pre-synaptic spike raster (1, num_steps)
    num_steps : int
        Number of simulation steps
    spike_prob : List[float]
        Spike probabilities for [neuron A, neuron B]
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    spike_raster_post : np.ndarray
        Post-synaptic spike raster (2, num_steps)
    """
    pre_synaptic_spikes = np.where(pre_spike_times.flatten() == 1)[0]

    spike_raster_post = np.zeros((2, num_steps))

    # Neuron A: tends to spike AFTER pre (within 20 steps) -> potentiation
    for ts in range(num_steps):
        for pre_ts in pre_synaptic_spikes:
            if ts in range(pre_ts, min(pre_ts + 20, num_steps)):
                if rng.random() < spike_prob[0]:
                    spike_raster_post[0, ts] = 1

    # Neuron B: tends to spike BEFORE pre (within 2-12 steps before) -> depression
    for ts in range(num_steps):
        for pre_ts in pre_synaptic_spikes:
            if ts in range(max(0, pre_ts - 12), max(0, pre_ts - 2)):
                if rng.random() < spike_prob[1]:
                    spike_raster_post[1, ts] = 1

    return spike_raster_post


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_rstdp(
    num_steps: int = 200,
    rate_pre: float = 0.03,
    rate_post: Tuple[float, float] = (0.09, 0.09),
    threshold: float = 1.0,
    spike_fraction: float = 0.4,
    dv: float = 0.04,
    du: float = 1.0,
    bias: float = 0.0,
    seed: int = 0,
    # R-STDP parameters (normalized to unit scale)
    learning_rate: float = 0.1,
    A_plus: float = 1.0,
    A_minus: float = -1.0,
    pre_trace_tau: float = 20.0,
    post_trace_tau: float = 20.0,
    pre_trace_impulse: float = 1.0,
    post_trace_impulse: float = 1.0,
    eligibility_tau: float = 0.05,  # Decay factor, NOT time constant! Effective τ ≈ 1/0.05 = 20 steps
    t_epoch: int = 1,
    # Reward generation (random windows)
    reward_prob: float = 0.02,
    reward_min_len: int = 7,
    reward_max_len: int = 30,
    reward_amplitude: float = 1.0,
    # Weight management
    w_init: float = 0.5,
    w_min: float = 0.0,
    w_max: float = 1.0,
    # State management
    use_continuous_state: bool = False,
    state: Optional[SimulationState] = None,
) -> Dict[str, object]:
    """
    Simulate R-STDP network: 1 pre-synaptic LIF neuron, 2 post-synaptic RSTDPLIF neurons.

    The network demonstrates reward-modulated STDP:
    - Neuron A receives reward in [reward_A_start, reward_A_end]
    - Neuron B receives reward in [reward_B_start, reward_B_end]
    - Weights change only when reward is present

    All values are normalized to ~[0, 1] range:
    - Threshold = 1.0
    - Traces in [0, 1] (approximately)
    - Weights clipped to [w_min, w_max]

    Returns
    -------
    Dict with neurons data, R-STDP traces, and state
    """
    # ═══════════════════════════════════════════════════════════════════════
    # STATE INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════

    if state is not None:
        sim_state = state
    elif use_continuous_state:
        sim_state = get_or_create_state(w_init, w_min, w_max)
    else:
        sim_state = SimulationState(
            w_A=w_init,
            w_B=w_init,
            w_min=w_min,
            w_max=w_max,
        )

    sim_state.clip_weights()

    rng = np.random.default_rng(seed + sim_state.seed_counter)
    spike_amp = threshold * spike_fraction

    # ═══════════════════════════════════════════════════════════════════════
    # INPUT GENERATION
    # ═══════════════════════════════════════════════════════════════════════

    # Pre-synaptic input spikes (single pre neuron)
    spike_raster_pre = (rng.random((1, num_steps)) < rate_pre).astype(np.int16)

    # Post-synaptic input spikes (correlated with pre)
    spike_raster_post = generate_post_spikes(
        spike_raster_pre, num_steps, list(rate_post), rng
    ).astype(np.int16)

    # Graded reward spikes for neurons A and B
    # Random windows: amplitude=reward_amplitude, length in [reward_min_len, reward_max_len],
    # with probability reward_prob to start a new window when currently inactive.
    graded_reward = np.zeros((2, num_steps), dtype=np.float32)
    reward_min_len = max(1, int(reward_min_len))
    reward_max_len = max(reward_min_len, int(reward_max_len))
    reward_prob = float(reward_prob)

    for neuron_idx in range(2):
        t = 0
        while t < num_steps:
            # Если уже активна награда, просто двигаемся дальше (окно уже проставлено)
            if graded_reward[neuron_idx, t] > 0:
                t += 1
                continue

            # Попытка запустить новое окно награды
            if rng.random() < reward_prob:
                dur = int(rng.integers(reward_min_len, reward_max_len + 1))
                end = min(num_steps, t + dur)
                graded_reward[neuron_idx, t:end] = reward_amplitude
                t = end
            else:
                t += 1

    # ═══════════════════════════════════════════════════════════════════════
    # R-STDP LEARNING RULE (normalized parameters)
    # ═══════════════════════════════════════════════════════════════════════

    rstdp = RewardModulatedSTDP(
        learning_rate=learning_rate,
        A_plus=A_plus,
        A_minus=A_minus,
        pre_trace_decay_tau=pre_trace_tau,
        post_trace_decay_tau=post_trace_tau,
        pre_trace_kernel_magnitude=pre_trace_impulse,
        post_trace_kernel_magnitude=post_trace_impulse,
        eligibility_trace_decay_tau=eligibility_tau,
        t_epoch=t_epoch,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # PROCESS CREATION
    # ═══════════════════════════════════════════════════════════════════════

    # Input sources
    pattern_pre = SpikeIn(data=spike_raster_pre.astype(int))
    pattern_post = SpikeIn(data=spike_raster_post.astype(int))
    reward_pattern = SpikeIn(data=graded_reward)

    # Input connectivity (scale spikes to reach threshold)
    conn_inp_pre = Dense(weights=np.array([[spike_amp]]))
    conn_inp_post = Dense(weights=np.eye(2) * spike_amp)
    # Graded reward connection (num_message_bits > 0 for graded spikes)
    conn_inp_reward = Dense(weights=np.eye(2), num_message_bits=8)

    # Pre-synaptic LIF neuron
    lif_pre = LIF(
        shape=(1,),
        u=sim_state.u_pre,
        v=sim_state.v_pre,
        du=du,
        dv=dv,
        vth=threshold,
        bias_mant=bias,
        name="lif_pre",
    )

    # Plastic connections (pre -> post A, pre -> post B)
    # Initialize with stored weights
    plast_conn = LearningDense(
        weights=np.array([[sim_state.w_A], [sim_state.w_B]]),
        learning_rule=rstdp,
        name="plastic_dense",
    )

    # Post-synaptic RSTDPLIF neurons (2 neurons for A and B)
    lif_post = RSTDPLIF(
        shape=(2,),
        u=np.array([sim_state.u_post_A, sim_state.u_post_B]),
        v=np.array([sim_state.v_post_A, sim_state.v_post_B]),
        du=du,
        dv=dv,
        vth=threshold,
        bias_mant=bias,
        name="lif_post",
        learning_rule=rstdp,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # MONITORING (using Read processes instead of Monitor)
    # ═══════════════════════════════════════════════════════════════════════

    spike_sink_pre = SinkRing(shape=(1,), buffer=num_steps)
    spike_sink_post = SinkRing(shape=(2,), buffer=num_steps)

    v_reader_pre = Read(buffer=num_steps, interval=1, offset=0)
    v_reader_post = Read(buffer=num_steps, interval=1, offset=0)

    w_reader = Read(buffer=num_steps, interval=1, offset=0)
    tag_reader = Read(buffer=num_steps, interval=1, offset=0)
    x1_reader = Read(buffer=num_steps, interval=1, offset=0)
    y1_reader = Read(buffer=num_steps, interval=1, offset=0)

    v_reader_pre.connect_var(lif_pre.v)
    v_reader_post.connect_var(lif_post.v)
    w_reader.connect_var(plast_conn.weights)
    tag_reader.connect_var(plast_conn.tag_1)
    x1_reader.connect_var(plast_conn.x1)
    y1_reader.connect_var(plast_conn.y1)

    # ═══════════════════════════════════════════════════════════════════════
    # NETWORK TOPOLOGY
    # ═══════════════════════════════════════════════════════════════════════

    # Input connections
    pattern_pre.s_out.connect(conn_inp_pre.s_in)
    conn_inp_pre.a_out.connect(lif_pre.a_in)

    pattern_post.s_out.connect(conn_inp_post.s_in)
    conn_inp_post.a_out.connect(lif_post.a_in)

    # Reward connections
    reward_pattern.s_out.connect(conn_inp_reward.s_in)
    conn_inp_reward.a_out.connect(lif_post.a_third_factor_in)

    # Plastic synapse connections
    lif_pre.s_out.connect(plast_conn.s_in)
    plast_conn.a_out.connect(lif_post.a_in)

    # CRITICAL: Learning trace connections from post neurons
    lif_post.s_out_bap.connect(plast_conn.s_in_bap)
    lif_post.s_out_y1.connect(plast_conn.s_in_y1)
    lif_post.s_out_y2.connect(plast_conn.s_in_y2)
    lif_post.s_out_y3.connect(plast_conn.s_in_y3)

    # Spike sinks
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
    raw_s_pre = np.array(spike_sink_pre.data.get()).astype(int)
    raw_s_post = np.array(spike_sink_post.data.get()).astype(int)
    raw_w = np.array(w_reader.data.get())
    raw_tag = np.array(tag_reader.data.get())
    raw_x1 = np.array(x1_reader.data.get())
    raw_y1 = np.array(y1_reader.data.get())

    lif_pre.stop()

    # ═══════════════════════════════════════════════════════════════════════
    # PROCESS DATA
    # ═══════════════════════════════════════════════════════════════════════

    def pad_array(arr, length, fill_value=0):
        if len(arr) >= length:
            return arr[:length]
        return np.pad(arr, (0, length - len(arr)), constant_values=fill_value)

    # Membrane potentials
    v_pre = pad_array(raw_v_pre.flatten(), num_steps)
    v_post_flat = raw_v_post.reshape(-1, num_steps) if raw_v_post.size >= num_steps * 2 else np.zeros((2, num_steps))
    v_post_A = v_post_flat[0] if v_post_flat.shape[0] > 0 else np.zeros(num_steps)
    v_post_B = v_post_flat[1] if v_post_flat.shape[0] > 1 else np.zeros(num_steps)

    # Spikes
    s_pre = pad_array(raw_s_pre.flatten(), num_steps)
    s_post_flat = raw_s_post.reshape(-1, num_steps) if raw_s_post.size >= num_steps * 2 else np.zeros((2, num_steps))
    s_post_A = s_post_flat[0] if s_post_flat.shape[0] > 0 else np.zeros(num_steps)
    s_post_B = s_post_flat[1] if s_post_flat.shape[0] > 1 else np.zeros(num_steps)

    # Weights (shape: num_steps x 2 x 1 -> extract w_A and w_B time series)
    if raw_w.size >= num_steps * 2:
        w_data = raw_w.reshape(num_steps, 2, 1)
        w_A_history = w_data[:, 0, 0]
        w_B_history = w_data[:, 1, 0]
    else:
        w_A_history = np.full(num_steps, sim_state.w_A)
        w_B_history = np.full(num_steps, sim_state.w_B)

    # Eligibility traces (tag_1, shape: num_steps x 2 x 1)
    if raw_tag.size >= num_steps * 2:
        tag_data = raw_tag.reshape(num_steps, 2, 1)
        eligibility_A = tag_data[:, 0, 0]
        eligibility_B = tag_data[:, 1, 0]
    else:
        eligibility_A = np.zeros(num_steps)
        eligibility_B = np.zeros(num_steps)

    # Pre-synaptic trace (x1)
    pre_trace = pad_array(raw_x1.flatten(), num_steps)

    # Clip weights to bounds
    w_A_history = np.clip(w_A_history, sim_state.w_min, sim_state.w_max)
    w_B_history = np.clip(w_B_history, sim_state.w_min, sim_state.w_max)

    # ═══════════════════════════════════════════════════════════════════════
    # UPDATE STATE FOR CONTINUITY
    # ═══════════════════════════════════════════════════════════════════════

    sim_state.v_pre = float(v_pre[-1]) if len(v_pre) > 0 else 0.0
    sim_state.v_post_A = float(v_post_A[-1]) if len(v_post_A) > 0 else 0.0
    sim_state.v_post_B = float(v_post_B[-1]) if len(v_post_B) > 0 else 0.0

    sim_state.w_A = float(w_A_history[-1]) if len(w_A_history) > 0 else sim_state.w_A
    sim_state.w_B = float(w_B_history[-1]) if len(w_B_history) > 0 else sim_state.w_B

    sim_state.pre_trace = float(pre_trace[-1]) if len(pre_trace) > 0 else 0.0
    sim_state.eligibility_A = float(eligibility_A[-1]) if len(eligibility_A) > 0 else 0.0
    sim_state.eligibility_B = float(eligibility_B[-1]) if len(eligibility_B) > 0 else 0.0

    # Store last reward values from input (for continuity visualization)
    sim_state.reward_A = float(graded_reward[0, -1])
    sim_state.reward_B = float(graded_reward[1, -1])

    sim_state.clip_weights()
    sim_state.seed_counter += 1

    # ═══════════════════════════════════════════════════════════════════════
    # POST TRACES FROM MODEL (read real y1 from plast_conn)
    # ═══════════════════════════════════════════════════════════════════════
    
    # Shape of raw_y1: (num_steps, 2, 1) after reshape
    if raw_y1.size >= num_steps * 2:
        y1_data = raw_y1.reshape(num_steps, 2, 1)
        post_trace_A = y1_data[:, 0, 0]
        post_trace_B = y1_data[:, 1, 0]
    else:
        post_trace_A = np.zeros(num_steps)
        post_trace_B = np.zeros(num_steps)

    sim_state.post_trace_A = float(post_trace_A[-1]) if len(post_trace_A) > 0 else 0.0
    sim_state.post_trace_B = float(post_trace_B[-1]) if len(post_trace_B) > 0 else 0.0

    # ═══════════════════════════════════════════════════════════════════════
    # FORMAT OUTPUT
    # ═══════════════════════════════════════════════════════════════════════

    neurons: List[Dict[str, object]] = [
        {
            "name": "Pre",
            "input_any": (spike_raster_pre.sum(axis=0) > 0).astype(int).tolist(),
            "membrane_potential": v_pre.tolist(),
            "spikes": s_pre.tolist(),
        },
        {
            "name": "Post A",
            "input_any": ((spike_raster_post[0] > 0) | (s_pre > 0)).astype(int).tolist(),
            "membrane_potential": v_post_A.tolist(),
            "spikes": s_post_A.astype(int).tolist(),
        },
        {
            "name": "Post B",
            "input_any": ((spike_raster_post[1] > 0) | (s_pre > 0)).astype(int).tolist(),
            "membrane_potential": v_post_B.tolist(),
            "spikes": s_post_B.astype(int).tolist(),
        },
    ]

    return {
        "dt": 1,
        "threshold": threshold,
        "dv": dv,
        "spike_amplitude": float(spike_amp),
        "neurons": neurons,
        "inputs_detail": [
            spike_raster_pre.astype(int).tolist(),
            spike_raster_post[0:1].astype(int).tolist(),
            spike_raster_post[1:2].astype(int).tolist(),
        ],
        "rstdp": {
            "pre_trace": pre_trace.tolist(),
            "post_trace_A": post_trace_A.tolist(),
            "post_trace_B": post_trace_B.tolist(),
            "eligibility_A": eligibility_A.tolist(),
            "eligibility_B": eligibility_B.tolist(),
            "reward_A": graded_reward[0].tolist(),
            "reward_B": graded_reward[1].tolist(),
            "weight_A": w_A_history.tolist(),
            "weight_B": w_B_history.tolist(),
            "w_min": sim_state.w_min,
            "w_max": sim_state.w_max,
        },
        "state": sim_state.to_dict(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate R-STDP network (1 pre, 2 post neurons) with lava-nc."
    )
    parser.add_argument("--steps", type=int, default=200, help="Simulation steps")
    parser.add_argument("--rate-pre", type=float, default=0.03, help="Pre-synaptic spike probability")
    parser.add_argument("--rate-post-a", type=float, default=0.09, help="Post A spike probability")
    parser.add_argument("--rate-post-b", type=float, default=0.09, help="Post B spike probability")
    parser.add_argument("--threshold", type=float, default=1.0, help="Spike threshold")
    parser.add_argument("--spike-fraction", type=float, default=0.4, help="Input spike amplitude as fraction of threshold")
    parser.add_argument("--dv", type=float, default=0.04, help="Membrane leak factor")

    # R-STDP parameters
    parser.add_argument("--learning-rate", type=float, default=0.1, help="R-STDP learning rate")
    parser.add_argument("--pre-trace-tau", type=float, default=10.0, help="Pre-synaptic trace decay tau")
    parser.add_argument("--post-trace-tau", type=float, default=10.0, help="Post-synaptic trace decay tau")
    parser.add_argument("--eligibility-tau", type=float, default=0.05, help="Eligibility decay factor (NOT tau! Effective τ ≈ 1/value)")

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

    parser.add_argument("--seed", type=int, default=156, help="RNG seed (156 matches tutorial)")
    parser.add_argument("--output", type=Path, default=Path("static/data.json"), help="Output JSON path")
    parser.add_argument("--continuous", action="store_true", help="Use continuous state")
    args = parser.parse_args()

    traces = simulate_rstdp(
        num_steps=args.steps,
        rate_pre=args.rate_pre,
        rate_post=(args.rate_post_a, args.rate_post_b),
        threshold=args.threshold,
        spike_fraction=args.spike_fraction,
        dv=args.dv,
        seed=args.seed,
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
        use_continuous_state=args.continuous,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved R-STDP traces -> {args.output}")
    print(f"Final weights: w_A={traces['state']['w_A']:.4f}, w_B={traces['state']['w_B']:.4f}")
    print(f"Weight bounds: [{traces['rstdp']['w_min']:.2f}, {traces['rstdp']['w_max']:.2f}]")


if __name__ == "__main__":
    main()
