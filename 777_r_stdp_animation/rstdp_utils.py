import numpy as np

from lava.proc.lif.process import LearningLIF
from lava.magma.core.model.py.neuron import LearningNeuronModelFloat
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.proc.lif.models import AbstractPyLifModelFloat


class RSTDPLIF(LearningLIF):
    """Learning LIF neuron used for R-STDP in floating point."""

    pass


@implements(proc=RSTDPLIF, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class RSTDPLIFModelFloat(LearningNeuronModelFloat, AbstractPyLifModelFloat):
    """Floating-point R-STDP LIF neuron model.

    Extends LearningLIF with:
    - custom post-synaptic trace y1
    - third-factor reward trace y2 driven by graded input
    - BAP and trace outputs for LearningDense.
    """

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    vth: float = LavaPyType(float, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.s_out_buff = np.zeros(proc_params["shape"])

    def spiking_activation(self):
        """Spiking activation function for Learning LIF."""
        return self.v > self.vth

    def calculate_third_factor_trace(self, s_graded_in: float) -> float:
        """Convert graded spikes into a third-factor reward trace.

        In this simple example the reward trace equals the graded input.
        """
        return s_graded_in

    def compute_post_synaptic_trace(self, s_out_buff):
        """Compute post-synaptic trace values for this time step."""
        y1_tau = self._learning_rule.post_trace_decay_tau
        y1_impulse = self._learning_rule.post_trace_kernel_magnitude

        return self.y1 * np.exp(-1.0 / y1_tau) + y1_impulse * s_out_buff

    def run_spk(self) -> None:
        """Update traces and send BAP / third-factor outputs."""

        # Update post-synaptic trace y1 based on previous trace and spikes.
        self.y1 = self.compute_post_synaptic_trace(self.s_out_buff)

        # Run standard LearningLIF dynamics (updates v, s_out, etc.).
        super().run_spk()

        # Receive graded reward input.
        a_graded_in = self.a_third_factor_in.recv()

        # Compute third-factor reward trace.
        self.y2 = self.calculate_third_factor_trace(a_graded_in)

        # Send BAP and traces to LearningDense.
        self.s_out_bap.send(self.s_out_buff)
        self.s_out_y1.send(self.y1)
        self.s_out_y2.send(self.y2)
        self.s_out_y3.send(self.y3)


def generate_post_spikes(pre_spike_times: np.ndarray,
                         num_steps: int,
                         spike_prob_post: np.ndarray) -> np.ndarray:
    """Generate post-synaptic spikes for two neurons to demonstrate
    potentiation and depression patterns used in the R-STDP tutorial.

    Parameters
    ----------
    pre_spike_times : ndarray
        Binary spike raster of the pre-synaptic neuron (shape: [1, T]).
    num_steps : int
        Number of simulation time steps.
    spike_prob_post : ndarray
        Two-element array with probabilities for the two post-synaptic
        spike trains.
    """
    pre_synaptic_spikes = np.where(pre_spike_times == 1)[1]

    spike_raster_post = np.zeros((len(spike_prob_post), num_steps))

    # Potentiation window for neuron A
    for ts in range(num_steps):
        for pre_ts in pre_synaptic_spikes:
            if ts in range(pre_ts, pre_ts + 20):
                if np.random.rand(1) < spike_prob_post[0]:
                    spike_raster_post[0][ts] = 1

    # Depression window for neuron B
    for ts in range(num_steps):
        for pre_ts in pre_synaptic_spikes:
            if ts in range(pre_ts - 12, pre_ts - 2):
                if np.random.rand(1) < spike_prob_post[1]:
                    spike_raster_post[1][ts] = 1

    return spike_raster_post


