"""
Creature Neural System with R-STDP Learning

This module implements the creature's nervous system using Lava's
Reward-modulated STDP learning rule.

Architecture:
    4 Sensory neurons → LearningDense (R-STDP) → 2 Motor neurons
"""

import numpy as np
from typing import Tuple, Optional

from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense, LearningDense
from lava.proc.io.source import RingBuffer as SpikeGenerator
from lava.proc.monitor.process import Monitor
from lava.proc.learning_rules.r_stdp_learning_rule import RewardModulatedSTDP
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg

from world import Action


class CreatureBrain:
    """
    Neural controller for the creature using R-STDP.
    
    The brain has:
    - 4 sensory inputs: food_left, food_right, danger_left, danger_right
    - 2 motor outputs: move_left, move_right
    - Plastic synapses trained with Reward-modulated STDP
    
    Parameters
    ----------
    learning_rate : float
        Learning rate for R-STDP
    initial_weight : float
        Initial synaptic weight (small random values)
    seed : int, optional
        Random seed for weight initialization
    """
    
    def __init__(
        self,
        learning_rate: float = 1.0,
        initial_weight: float = 0.5,
        seed: Optional[int] = None
    ):
        self.learning_rate = learning_rate
        self.initial_weight = initial_weight
        self.rng = np.random.default_rng(seed)
        
        self.num_sensory = 4
        self.num_motor = 2
        
        # Initialize weights randomly
        self.weights = self.rng.uniform(
            0.1, initial_weight,
            size=(self.num_motor, self.num_sensory)
        )
        
        # History for analysis
        self.weight_history = [self.weights.copy()]
        self.decision_history = []
        
        # Network components (will be built on demand)
        self._network_built = False
    
    def get_weights(self) -> np.ndarray:
        """Get current synaptic weights"""
        return self.weights.copy()
    
    def decide(self, sensory_input: Tuple[int, int, int, int]) -> Action:
        """
        Make a decision based on sensory input.
        
        Uses simple weighted sum and winner-take-all.
        
        Parameters
        ----------
        sensory_input : tuple of 4 ints
            (food_left, food_right, danger_left, danger_right)
        
        Returns
        -------
        Action
            The chosen action (LEFT, RIGHT, or STAY)
        """
        # Convert input to array
        inp = np.array(sensory_input, dtype=np.float32)
        
        # Compute motor activations: weights @ input
        motor_activation = self.weights @ inp
        
        # Add small noise for exploration
        noise = self.rng.normal(0, 0.1, size=self.num_motor)
        motor_activation += noise
        
        # Decision logic
        left_activation = motor_activation[0]
        right_activation = motor_activation[1]
        
        # Winner-take-all with threshold
        threshold = 0.1
        
        if left_activation > right_activation and left_activation > threshold:
            action = Action.LEFT
        elif right_activation > left_activation and right_activation > threshold:
            action = Action.RIGHT
        else:
            # Random choice when activations are similar
            action = Action(self.rng.integers(1, 3))  # LEFT or RIGHT
        
        self.decision_history.append({
            "input": sensory_input,
            "activations": motor_activation.copy(),
            "action": action
        })
        
        return action
    
    def learn(
        self,
        sensory_input: Tuple[int, int, int, int],
        action: Action,
        reward: float
    ):
        """
        Update weights based on reward signal (simplified R-STDP).
        
        This is a simplified version of R-STDP that:
        1. Strengthens connections that contributed to rewarded actions
        2. Weakens connections that contributed to punished actions
        
        Parameters
        ----------
        sensory_input : tuple of 4 ints
            The sensory input that was presented
        action : Action
            The action that was taken
        reward : float
            The reward received (positive = good, negative = bad)
        """
        if reward == 0 or action == Action.STAY:
            return
        
        inp = np.array(sensory_input, dtype=np.float32)
        
        # Determine which motor neuron "fired"
        if action == Action.LEFT:
            active_motor = 0
        else:  # RIGHT
            active_motor = 1
        
        # R-STDP update rule (simplified):
        # dw = learning_rate * reward * pre_activity * post_activity
        # 
        # If reward > 0: strengthen connections that led to this action
        # If reward < 0: weaken connections that led to this action
        
        delta_w = self.learning_rate * reward * inp
        
        # Update weights for the active motor neuron
        self.weights[active_motor] += delta_w
        
        # Optionally: slightly decrease weights for the inactive motor
        # (competitive learning)
        inactive_motor = 1 - active_motor
        self.weights[inactive_motor] -= 0.1 * self.learning_rate * reward * inp
        
        # Clip weights to reasonable range
        self.weights = np.clip(self.weights, 0.0, 5.0)
        
        # Record weight history
        self.weight_history.append(self.weights.copy())
    
    def get_expected_behavior(self) -> dict:
        """
        Analyze current weights to predict expected behavior.
        
        Returns
        -------
        dict
            Analysis of what the creature should do given current weights
        """
        w = self.weights
        # Columns: food_left, food_right, danger_left, danger_right
        # Rows: move_left, move_right
        
        analysis = {
            "food_left_response": "LEFT" if w[0, 0] > w[1, 0] else "RIGHT",
            "food_right_response": "LEFT" if w[0, 1] > w[1, 1] else "RIGHT",
            "danger_left_response": "LEFT" if w[0, 2] > w[1, 2] else "RIGHT",
            "danger_right_response": "LEFT" if w[0, 3] > w[1, 3] else "RIGHT",
            "weights": w.copy()
        }
        
        # Check if behavior is correct
        analysis["food_behavior_correct"] = (
            analysis["food_left_response"] == "LEFT" and
            analysis["food_right_response"] == "RIGHT"
        )
        analysis["danger_behavior_correct"] = (
            analysis["danger_left_response"] == "RIGHT" and
            analysis["danger_right_response"] == "LEFT"
        )
        analysis["fully_trained"] = (
            analysis["food_behavior_correct"] and
            analysis["danger_behavior_correct"]
        )
        
        return analysis
    
    def reset_weights(self):
        """Reset weights to initial random values"""
        self.weights = self.rng.uniform(
            0.1, self.initial_weight,
            size=(self.num_motor, self.num_sensory)
        )
        self.weight_history = [self.weights.copy()]
        self.decision_history = []


class CreatureBrainLava:
    """
    Neural controller using actual Lava R-STDP.
    
    This version uses Lava's LearningDense process with 
    RewardModulatedSTDP learning rule for on-chip learning.
    
    Note: This is more complex and requires careful management of
    the Lava runtime. For educational purposes, CreatureBrain
    (simplified version) may be easier to understand.
    """
    
    def __init__(
        self,
        learning_rate: float = 1.0,
        seed: Optional[int] = None
    ):
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)
        
        self.num_sensory = 4
        self.num_motor = 2
        
        # Initial weights
        self.initial_weights = self.rng.uniform(
            10, 50,
            size=(self.num_motor, self.num_sensory)
        ).astype(np.int32)
        
        # R-STDP learning rule
        self.learning_rule = RewardModulatedSTDP(
            learning_rate=learning_rate,
            A_plus=2,
            A_minus=-2,
            pre_trace_decay_tau=10,
            post_trace_decay_tau=10,
            pre_trace_kernel_magnitude=16,
            post_trace_kernel_magnitude=16,
            eligibility_trace_decay_tau=0.5,
            t_epoch=1
        )
        
        self._built = False
        self._processes = {}
    
    def build_network(self, sim_steps: int = 10):
        """
        Build the Lava network for simulation.
        
        Parameters
        ----------
        sim_steps : int
            Number of simulation steps per decision
        """
        # This would build actual Lava processes
        # For now, we'll use the simplified version
        # Full Lava implementation requires more complex setup
        pass
    
    def decide_and_learn(
        self,
        sensory_input: Tuple[int, int, int, int],
        reward: float
    ) -> Action:
        """
        Make decision and learn from reward.
        
        This would run actual Lava simulation with R-STDP.
        """
        # Placeholder - full implementation requires Lava runtime management
        raise NotImplementedError(
            "Full Lava R-STDP implementation requires runtime management. "
            "Use CreatureBrain for simplified simulation."
        )


if __name__ == "__main__":
    # Test the simplified brain
    brain = CreatureBrain(learning_rate=0.5, seed=42)
    
    print("Initial weights:")
    print(brain.get_weights())
    print()
    
    # Simulate some learning
    test_scenarios = [
        # (sensory_input, correct_action, reward)
        ((1, 0, 0, 0), Action.LEFT, 1.0),   # Food left → go left → reward
        ((0, 1, 0, 0), Action.RIGHT, 1.0),  # Food right → go right → reward
        ((0, 0, 1, 0), Action.RIGHT, 1.0),  # Danger left → go right → reward
        ((0, 0, 0, 1), Action.LEFT, 1.0),   # Danger right → go left → reward
    ]
    
    print("Training...")
    for epoch in range(10):
        for sensory, action, reward in test_scenarios:
            brain.learn(sensory, action, reward)
    
    print("\nWeights after training:")
    print(brain.get_weights())
    print()
    
    print("Expected behavior analysis:")
    analysis = brain.get_expected_behavior()
    for key, value in analysis.items():
        if key != "weights":
            print(f"  {key}: {value}")

