"""
Creature Neural System with R-STDP Learning

This module implements the creature's nervous system using 
Reward-modulated STDP learning rule with eligibility traces.

Architectures:
    Simple:  4 Sensory → 2 Motor (8 synapses)
    Deep:    4 Sensory → 6 Hidden1 → 4 Hidden2 → 2 Motor (56 synapses)
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

from world import Action


@dataclass
class LayerState:
    """State of a single layer in the network"""
    weights: np.ndarray           # Synaptic weights
    eligibility: np.ndarray       # Eligibility traces for R-STDP
    activation: np.ndarray        # Current activation values
    input_cache: np.ndarray       # Cached input for learning
    trainable: bool = True        # Whether weights can change


class DeepCreatureBrain:
    """
    Deep Neural controller with eligibility traces for R-STDP.
    
    Architecture: 4 Sensory → Hidden1 → Hidden2 → 2 Motor
    
    Features:
    - Multiple plastic layers with eligibility traces
    - Reward signal broadcasts to all layers
    - Weight decay (forgetting) when not reinforced
    - Competitive learning between motor neurons
    - Exploration annealing (epsilon decreases over time)
    
    Parameters
    ----------
    hidden_sizes : list of int
        Sizes of hidden layers (default: [6, 4])
    learning_rate : float
        Base learning rate for R-STDP
    trace_decay : float
        Decay rate for eligibility traces (0-1)
    weight_decay : float
        Weight decay rate per step (forgetting)
    initial_weight : float
        Maximum initial weight value
    epsilon_start : float
        Initial exploration rate (random action probability)
    epsilon_end : float
        Final exploration rate after annealing
    epsilon_decay : float
        Decay rate for epsilon (per decision)
    seed : int, optional
        Random seed
    """
    
    def __init__(
        self,
        hidden_sizes: List[int] = None,
        learning_rate: float = 3.0,  # Increased for stronger plasticity
        trace_decay: float = 0.7,    # Reduced for sharper traces
        weight_decay: float = 0.0001,
        initial_weight: float = 0.5,
        epsilon_start: float = 0.3,  # 30% random at start
        epsilon_end: float = 0.02,   # 2% random at end
        epsilon_decay: float = 0.995,  # Slow decay
        seed: Optional[int] = None
    ):
        if hidden_sizes is None:
            hidden_sizes = [6, 4]
        
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.trace_decay = trace_decay
        self.weight_decay = weight_decay
        self.initial_weight = initial_weight
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        
        # Network structure
        self.num_sensory = 4
        self.num_motor = 2
        self.layer_sizes = [self.num_sensory] + hidden_sizes + [self.num_motor]
        
        # Build layers
        self.layers: List[LayerState] = []
        self._build_network()
        
        # History
        self.weight_history = []
        self.eligibility_history = []
        self.decision_history = []
        
        self._record_state()
    
    def _build_network(self):
        """Initialize all layers with random weights"""
        self.layers = []
        
        last_layer_index = len(self.layer_sizes) - 2

        for i in range(len(self.layer_sizes) - 1):
            in_size = self.layer_sizes[i]
            out_size = self.layer_sizes[i + 1]
            
            if i == last_layer_index:
                # Final motor layer is fixed: pass-through matrix of ones
                weights = np.ones((out_size, in_size), dtype=np.float32)
                trainable = False
            else:
                # Xavier initialization for plastic layers
                scale = np.sqrt(2.0 / (in_size + out_size))
                weights = self.rng.uniform(
                    0.1 * scale,
                    self.initial_weight * scale,
                    size=(out_size, in_size)
                )
                trainable = True
            
            layer = LayerState(
                weights=weights,
                eligibility=np.zeros((out_size, in_size)),
                activation=np.zeros(out_size),
                input_cache=np.zeros(in_size),
                trainable=trainable
            )
            self.layers.append(layer)
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def _record_state(self):
        """Record current state for visualization"""
        weights = [layer.weights.copy() for layer in self.layers]
        eligibilities = [layer.eligibility.copy() for layer in self.layers]
        self.weight_history.append(weights)
        self.eligibility_history.append(eligibilities)
    
    def forward(self, sensory_input: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Forward pass through the network.
        
        Also updates eligibility traces based on pre/post activity.
        
        Parameters
        ----------
        sensory_input : tuple of 4 ints
            (food_left, food_right, danger_left, danger_right)
        
        Returns
        -------
        np.ndarray
            Motor neuron activations
        """
        x = np.array(sensory_input, dtype=np.float32)
        
        for layer in self.layers:
            # Cache input for learning
            layer.input_cache = x.copy()
            
            # Forward: y = ReLU(W @ x)
            pre_activation = layer.weights @ x
            layer.activation = self._relu(pre_activation)
            
            # Update eligibility trace (Hebbian correlation)
            # e_ij = trace_decay * e_ij + post_i * pre_j
            layer.eligibility = (
                self.trace_decay * layer.eligibility +
                np.outer(layer.activation, x)
            )
            
            x = layer.activation
        
        return x
    
    def decide(self, sensory_input: Tuple[int, int, int, int]) -> Action:
        """
        Make a decision based on sensory input.
        
        Uses epsilon-greedy exploration with annealing.
        Includes STAY option when motor activations are below threshold.
        
        Parameters
        ----------
        sensory_input : tuple of 4 ints
        
        Returns
        -------
        Action
            LEFT, RIGHT, or STAY
        """
        motor_activation = self.forward(sensory_input)
        
        left_activation = motor_activation[0]
        right_activation = motor_activation[1]
        
        # Exploitation: winner-take-all with STAY threshold
        max_activation = max(left_activation, right_activation)
        min_activation = min(left_activation, right_activation)
        diff_activation = max_activation - min_activation

        tie_tolerance = 0.01 * max_activation
        
        action = Action.STAY
        if diff_activation <= tie_tolerance:
            action = Action.STAY
        elif left_activation > right_activation:
            action = Action.LEFT
        elif right_activation > left_activation:
            action = Action.RIGHT
            
        return action
    
    def learn(
        self,
        sensory_input: Tuple[int, int, int, int],
        action: Action,
        reward: float
    ):
        """
        Update weights using R-STDP with eligibility traces.
        
        The reward signal modulates ALL eligibility traces across all layers.
        This solves the credit assignment problem by using temporal correlation.
        
        Parameters
        ----------
        sensory_input : tuple of 4 ints
        action : Action
        reward : float
            Reward signal (+1 good, -1 bad, 0 neutral)
        """
        # Apply weight decay only to trainable layers (forgetting)
        for layer in self.layers:
            if layer.trainable:
                layer.weights *= (1.0 - self.weight_decay)
        
        if reward == 0 or action == Action.STAY:
            self._record_state()
            return
        
        # Determine active motor neuron
        active_motor = 0 if action == Action.LEFT else 1
        
        # Consider only trainable layers for learning
        trainable_layers = [layer for layer in self.layers if layer.trainable]
        num_layers = len(trainable_layers)
        if num_layers == 0:
            self._record_state()
            return
        
        for i, layer in enumerate(trainable_layers):
            # Decreasing learning rate for earlier layers
            layer_lr = self.learning_rate * (0.5 + 0.5 * (i + 1) / num_layers)
            
            # Clip eligibility to prevent explosion (tighter bounds for stability)
            layer.eligibility = np.clip(layer.eligibility, -5.0, 5.0)
            
            # R-STDP update: dW = lr * reward * eligibility
            if reward < 0:
                # Make punishments softer to avoid collapsing weights
                delta_w = layer_lr * reward * layer.eligibility * 0.3
            else:
                delta_w = layer_lr * reward * layer.eligibility
            
            # Apply update
            layer.weights += delta_w
            
            # Clip weights to valid range
            layer.weights = np.clip(layer.weights, 0.02, 3.0)
            
            # Partial reset of eligibility after reward
            layer.eligibility *= 0.3
        
        self._record_state()
    
    def get_weights(self) -> np.ndarray:
        """Get output layer weights (for compatibility)"""
        return self.layers[-1].weights.copy()
    
    def get_all_weights(self) -> List[np.ndarray]:
        """Get weights from all layers"""
        return [layer.weights.copy() for layer in self.layers]
    
    def get_all_eligibilities(self) -> List[np.ndarray]:
        """Get eligibility traces from all layers"""
        return [layer.eligibility.copy() for layer in self.layers]
    
    def get_layer_info(self) -> List[Dict]:
        """Get detailed info about each layer"""
        info = []
        for i, layer in enumerate(self.layers):
            info.append({
                "index": i,
                "input_size": layer.weights.shape[1],
                "output_size": layer.weights.shape[0],
                "num_synapses": layer.weights.size,
                "weights": layer.weights.copy(),
                "eligibility": layer.eligibility.copy(),
                "activation": layer.activation.copy(),
                "mean_weight": layer.weights.mean(),
                "max_weight": layer.weights.max(),
                "mean_eligibility": np.abs(layer.eligibility).mean()
            })
        return info
    
    def get_network_stats(self) -> Dict:
        """Get overall network statistics"""
        total_synapses = sum(layer.weights.size for layer in self.layers)
        total_eligibility = sum(np.abs(layer.eligibility).sum() for layer in self.layers)
        
        return {
            "num_layers": len(self.layers),
            "layer_sizes": self.layer_sizes,
            "total_synapses": total_synapses,
            "total_eligibility": total_eligibility,
            "architecture": " → ".join(str(s) for s in self.layer_sizes)
        }
    
    def get_expected_behavior(self) -> Dict:
        """Analyze expected behavior based on current weights"""
        # Test each sensory input pattern
        test_patterns = [
            ((1, 0, 0, 0), "food_left"),
            ((0, 1, 0, 0), "food_right"),
            ((0, 0, 1, 0), "danger_left"),
            ((0, 0, 0, 1), "danger_right"),
        ]
        
        analysis = {}
        
        for pattern, name in test_patterns:
            motor = self.forward(pattern)
            response = "LEFT" if motor[0] > motor[1] else "RIGHT"
            analysis[f"{name}_response"] = response
            analysis[f"{name}_activation"] = motor.copy()
        
        # Check correctness
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
        """Reset network to initial random state"""
        self._build_network()
        self.weight_history = []
        self.eligibility_history = []
        self.decision_history = []
        self._record_state()


# Keep original simple brain for comparison
class CreatureBrain:
    """
    Simple single-layer neural controller (original version).
    
    Architecture: 4 Sensory → 2 Motor (8 synapses)
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
        
        self.weights = self.rng.uniform(
            0.1, initial_weight,
            size=(self.num_motor, self.num_sensory)
        )
        
        self.weight_history = [self.weights.copy()]
        self.decision_history = []
        self._network_built = False
    
    def get_weights(self) -> np.ndarray:
        return self.weights.copy()
    
    def decide(self, sensory_input: Tuple[int, int, int, int]) -> Action:
        inp = np.array(sensory_input, dtype=np.float32)
        motor_activation = self.weights @ inp
        noise = self.rng.normal(0, 0.1, size=self.num_motor)
        motor_activation += noise
        
        left_activation = motor_activation[0]
        right_activation = motor_activation[1]
        threshold = 0.1
        
        if left_activation > right_activation and left_activation > threshold:
            action = Action.LEFT
        elif right_activation > left_activation and right_activation > threshold:
            action = Action.RIGHT
        else:
            action = Action(self.rng.integers(1, 3))
        
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
        if reward == 0 or action == Action.STAY:
            return
        
        inp = np.array(sensory_input, dtype=np.float32)
        active_motor = 0 if action == Action.LEFT else 1
        
        delta_w = self.learning_rate * reward * inp
        self.weights[active_motor] += delta_w
        
        inactive_motor = 1 - active_motor
        self.weights[inactive_motor] -= 0.1 * self.learning_rate * reward * inp
        
        self.weights = np.clip(self.weights, 0.0, 5.0)
        self.weight_history.append(self.weights.copy())
    
    def get_expected_behavior(self) -> Dict:
        w = self.weights
        
        analysis = {
            "food_left_response": "LEFT" if w[0, 0] > w[1, 0] else "RIGHT",
            "food_right_response": "LEFT" if w[0, 1] > w[1, 1] else "RIGHT",
            "danger_left_response": "LEFT" if w[0, 2] > w[1, 2] else "RIGHT",
            "danger_right_response": "LEFT" if w[0, 3] > w[1, 3] else "RIGHT",
            "weights": w.copy()
        }
        
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
        self.weights = self.rng.uniform(
            0.1, self.initial_weight,
            size=(self.num_motor, self.num_sensory)
        )
        self.weight_history = [self.weights.copy()]
        self.decision_history = []


if __name__ == "__main__":
    print("="*60)
    print("Testing DeepCreatureBrain with Eligibility Traces")
    print("="*60)
    
    # Create deep brain
    brain = DeepCreatureBrain(
        hidden_sizes=[6, 4],
        learning_rate=0.3,
        trace_decay=0.85,
        weight_decay=0.001,
        seed=42
    )
    
    stats = brain.get_network_stats()
    print(f"\nArchitecture: {stats['architecture']}")
    print(f"Total synapses: {stats['total_synapses']}")
    print(f"Number of layers: {stats['num_layers']}")
    
    print("\nLayer details:")
    for info in brain.get_layer_info():
        print(f"  Layer {info['index']}: {info['input_size']} → {info['output_size']} "
              f"({info['num_synapses']} synapses)")
    
    # Training scenarios
    test_scenarios = [
        ((1, 0, 0, 0), Action.LEFT, 1.0),   # Food left → go left
        ((0, 1, 0, 0), Action.RIGHT, 1.0),  # Food right → go right
        ((0, 0, 1, 0), Action.RIGHT, 1.0),  # Danger left → go right
        ((0, 0, 0, 1), Action.LEFT, 1.0),   # Danger right → go left
    ]
    
    print("\n" + "="*60)
    print("Training...")
    print("="*60)
    
    for epoch in range(20):
        for sensory, action, reward in test_scenarios:
            brain.learn(sensory, action, reward)
        
        if (epoch + 1) % 5 == 0:
            analysis = brain.get_expected_behavior()
            print(f"Epoch {epoch+1}: Fully trained = {analysis['fully_trained']}")
    
    print("\nFinal behavior analysis:")
    analysis = brain.get_expected_behavior()
    print(f"  Food left → {analysis['food_left_response']}")
    print(f"  Food right → {analysis['food_right_response']}")
    print(f"  Danger left → {analysis['danger_left_response']}")
    print(f"  Danger right → {analysis['danger_right_response']}")
    print(f"  Fully trained: {analysis['fully_trained']}")
