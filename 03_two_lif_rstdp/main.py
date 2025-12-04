"""Minimal two-neuron R-STDP demo built purely from lava-nc classes.
Run with: python projects/03_two_lif_rstdp/main.py
"""

import numpy as np

from lava.proc.lif.process import LIF, LearningLIF
from lava.proc.dense.process import Dense, LearningDense
from lava.proc.learning_rules.r_stdp_learning_rule import RewardModulatedSTDP
from lava.proc.io.source import RingBuffer as SpikeIn
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.model.py.neuron import (
    LearningNeuronModelFloat,
)
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.proc.lif.models import AbstractPyLifModelFloat


class RSTDPLIF(LearningLIF):
    """Learning LIF that exposes post traces and reward trace for R-STDP."""


@implements(proc=RSTDPLIF, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class RSTDPLIFModelFloat(LearningNeuronModelFloat, AbstractPyLifModelFloat):
    """Floating-point Learning LIF with third-factor trace."""

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    vth: float = LavaPyType(float, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.s_out_buff = np.zeros(proc_params["shape"])

    def spiking_activation(self):
        return self.v > self.vth

    def calculate_third_factor_trace(self, s_graded_in: float) -> float:
        return s_graded_in

    def compute_post_synaptic_trace(self, s_out_buff):
        y1_tau = self._learning_rule.post_trace_decay_tau
        y1_impulse = self._learning_rule.post_trace_kernel_magnitude
        return self.y1 * np.exp(-1 / y1_tau) + y1_impulse * s_out_buff

    def run_spk(self) -> None:
        self.y1 = self.compute_post_synaptic_trace(self.s_out_buff)
        super().run_spk()

        a_graded_in = self.a_third_factor_in.recv()
        self.y2 = self.calculate_third_factor_trace(a_graded_in)

        self.s_out_bap.send(self.s_out_buff)
        self.s_out_y1.send(self.y1)
        self.s_out_y2.send(self.y2)
        self.s_out_y3.send(self.y3)


def build_network(num_steps: int = 200, seed: int = 0):
    """Wire two LIF neurons with an R-STDP synapse and random reward."""
    rng = np.random.default_rng(seed)
    pre_spikes = (rng.random(num_steps) < 0.25).astype(float).reshape(1, num_steps)
    reward_signal = (rng.random(num_steps) < 0.12).astype(float).reshape(1, num_steps)

    learning_rule = RewardModulatedSTDP(
        learning_rate=0.2,
        A_plus=4.0,
        A_minus=-2.0,
        pre_trace_decay_tau=10.0,
        post_trace_decay_tau=10.0,
        pre_trace_kernel_magnitude=2.0,
        post_trace_kernel_magnitude=2.0,
        eligibility_trace_decay_tau=0.1,
        t_epoch=1,
    )

    pre = LIF(shape=(1,), du=0.0, dv=0.0, vth=0.5, bias_mant=0.0)
    syn = LearningDense(weights=np.array([[0.5]]), learning_rule=learning_rule)
    post = RSTDPLIF(
        shape=(1,),
        du=0.05,
        dv=0.01,
        vth=0.5,
        bias_mant=0.2,
        learning_rule=learning_rule,
    )

    input_spikes = SpikeIn(data=pre_spikes)
    reward = SpikeIn(data=reward_signal)
    reward_proj = Dense(weights=np.eye(1))

    input_spikes.s_out.connect(pre.a_in)
    pre.s_out.connect(syn.s_in)
    syn.a_out.connect(post.a_in)

    reward.s_out.connect(reward_proj.s_in)
    reward_proj.a_out.connect(post.a_third_factor_in)

    # Post-synaptic traces back to learning synapse
    post.s_out_bap.connect(syn.s_in_bap)
    post.s_out_y1.connect(syn.s_in_y1)
    post.s_out_y2.connect(syn.s_in_y2)
    post.s_out_y3.connect(syn.s_in_y3)

    return {
        "pre": pre,
        "post": post,
        "syn": syn,
        "input_spikes": pre_spikes,
        "reward_signal": reward_signal,
        "learning_rule": learning_rule,
    }


def run_network(num_steps: int = 200, seed: int = 0):
    net = build_network(num_steps=num_steps, seed=seed)
    pre = net["pre"]
    syn = net["syn"]

    run_cfg = Loihi2SimCfg(select_tag="floating_pt")
    run_condition = RunSteps(num_steps=num_steps)

    weight_before = syn.weights.get().copy()
    pre.run(condition=run_condition, run_cfg=run_cfg)
    weight_after = syn.weights.get().copy()
    pre.stop()

    net["weight_before"] = weight_before
    net["weight_after"] = weight_after
    return net


if __name__ == "__main__":
    steps = 300
    net = run_network(num_steps=steps, seed=42)

    print("Presynaptic spike times:", np.nonzero(net["input_spikes"][0])[0])
    print("Reward times:", np.nonzero(net["reward_signal"][0])[0])
    print("Initial weight:", net["weight_before"].ravel()[0])
    print("Final weight:", net["weight_after"].ravel()[0])
