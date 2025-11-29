"""
R-STDP Creature Simulation

Main simulation loop that connects the World and Creature's neural system.
Tracks learning progress and provides metrics.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from world import World, Action, EntityType
from creature import CreatureBrain


@dataclass
class EpisodeStats:
    """Statistics for a single episode"""
    total_reward: float = 0.0
    steps: int = 0
    food_approached: int = 0
    food_avoided: int = 0
    danger_approached: int = 0
    danger_avoided: int = 0
    correct_decisions: int = 0
    total_decisions: int = 0
    
    @property
    def accuracy(self) -> float:
        if self.total_decisions == 0:
            return 0.0
        return self.correct_decisions / self.total_decisions


@dataclass
class SimulationHistory:
    """Complete simulation history"""
    episode_stats: List[EpisodeStats] = field(default_factory=list)
    weight_snapshots: List[np.ndarray] = field(default_factory=list)
    step_log: List[Dict] = field(default_factory=list)


class Simulation:
    """
    Main simulation that trains a creature using R-STDP.
    
    Parameters
    ----------
    world : World
        The 1D world environment
    brain : CreatureBrain
        The creature's neural controller
    """
    
    def __init__(self, world: World, brain: CreatureBrain):
        self.world = world
        self.brain = brain
        self.history = SimulationHistory()
        self.current_episode_stats = EpisodeStats()
    
    def reset(self):
        """Reset simulation state"""
        self.world.reset()
        self.current_episode_stats = EpisodeStats()
    
    def step(self, learn: bool = True, verbose: bool = False) -> Dict:
        """
        Execute one simulation step.
        
        Parameters
        ----------
        learn : bool
            Whether to update weights based on reward
        verbose : bool
            Whether to print step information
        
        Returns
        -------
        dict
            Step information including action, reward, etc.
        """
        # Get sensory input
        sensory = self.world.get_sensory_input()
        
        # Brain decides action
        action = self.brain.decide(sensory)
        
        # Execute action in world
        state, reward, info = self.world.step(action)
        
        # Learn from reward
        if learn and reward != 0:
            self.brain.learn(sensory, action, reward)
        
        # Track statistics
        self._update_stats(sensory, action, reward)
        
        # Log step
        step_info = {
            "sensory": sensory,
            "action": action,
            "reward": reward,
            "creature_pos": state.creature_pos,
            "entity_type": state.entity_type,
            "entity_pos": state.entity_pos,
            "weights": self.brain.get_weights().copy()
        }
        self.history.step_log.append(step_info)
        
        if verbose:
            self._print_step(step_info)
        
        return step_info
    
    def _update_stats(self, sensory, action, reward):
        """Update episode statistics"""
        self.current_episode_stats.steps += 1
        self.current_episode_stats.total_reward += reward
        
        food_left, food_right, danger_left, danger_right = sensory
        
        # Track decisions when there's something to react to
        if any(sensory):
            self.current_episode_stats.total_decisions += 1
            
            # Check if decision was correct
            correct = False
            if food_left and action == Action.LEFT:
                self.current_episode_stats.food_approached += 1
                correct = True
            elif food_right and action == Action.RIGHT:
                self.current_episode_stats.food_approached += 1
                correct = True
            elif food_left and action == Action.RIGHT:
                self.current_episode_stats.food_avoided += 1
            elif food_right and action == Action.LEFT:
                self.current_episode_stats.food_avoided += 1
            elif danger_left and action == Action.RIGHT:
                self.current_episode_stats.danger_avoided += 1
                correct = True
            elif danger_right and action == Action.LEFT:
                self.current_episode_stats.danger_avoided += 1
                correct = True
            elif danger_left and action == Action.LEFT:
                self.current_episode_stats.danger_approached += 1
            elif danger_right and action == Action.RIGHT:
                self.current_episode_stats.danger_approached += 1
            
            if correct:
                self.current_episode_stats.correct_decisions += 1
    
    def _print_step(self, info: Dict):
        """Print step information"""
        sensory = info["sensory"]
        action = info["action"]
        reward = info["reward"]
        
        sensory_str = f"F_L={sensory[0]} F_R={sensory[1]} D_L={sensory[2]} D_R={sensory[3]}"
        print(f"  Sensory: {sensory_str}")
        print(f"  Action: {action.name}, Reward: {reward:+.1f}")
        print(self.world.render_ascii())
        print()
    
    def run_episode(
        self,
        num_steps: int = 100,
        learn: bool = True,
        verbose: bool = False
    ) -> EpisodeStats:
        """
        Run a complete episode.
        
        Parameters
        ----------
        num_steps : int
            Number of steps in the episode
        learn : bool
            Whether to enable learning
        verbose : bool
            Whether to print each step
        
        Returns
        -------
        EpisodeStats
            Statistics from this episode
        """
        self.reset()
        
        if verbose:
            print(f"\n{'='*50}")
            print("Starting Episode")
            print(f"{'='*50}\n")
            print("Initial state:")
            print(self.world.render_ascii())
            print()
        
        for step in range(num_steps):
            if verbose:
                print(f"Step {step + 1}:")
            self.step(learn=learn, verbose=verbose)
        
        # Save episode stats
        stats = self.current_episode_stats
        self.history.episode_stats.append(stats)
        self.history.weight_snapshots.append(self.brain.get_weights().copy())
        
        if verbose:
            print(f"\n{'='*50}")
            print("Episode Complete")
            print(f"Total Reward: {stats.total_reward:.1f}")
            print(f"Accuracy: {stats.accuracy:.1%}")
            print(f"{'='*50}\n")
        
        return stats
    
    def train(
        self,
        num_episodes: int = 50,
        steps_per_episode: int = 100,
        verbose_every: int = 10
    ) -> List[EpisodeStats]:
        """
        Train the creature over multiple episodes.
        
        Parameters
        ----------
        num_episodes : int
            Number of training episodes
        steps_per_episode : int
            Steps per episode
        verbose_every : int
            Print progress every N episodes (0 = never)
        
        Returns
        -------
        list of EpisodeStats
            Statistics from all episodes
        """
        print(f"Training for {num_episodes} episodes...")
        print(f"Initial weights:\n{self.brain.get_weights()}\n")
        
        all_stats = []
        
        for ep in range(num_episodes):
            verbose = (verbose_every > 0 and (ep + 1) % verbose_every == 0)
            
            if verbose:
                print(f"\n--- Episode {ep + 1}/{num_episodes} ---")
            
            stats = self.run_episode(
                num_steps=steps_per_episode,
                learn=True,
                verbose=False
            )
            all_stats.append(stats)
            
            if verbose:
                print(f"Reward: {stats.total_reward:.1f}, "
                      f"Accuracy: {stats.accuracy:.1%}, "
                      f"Decisions: {stats.total_decisions}")
        
        print(f"\nTraining complete!")
        print(f"Final weights:\n{self.brain.get_weights()}")
        
        # Analyze final behavior
        analysis = self.brain.get_expected_behavior()
        print(f"\nBehavior analysis:")
        print(f"  Food behavior correct: {analysis['food_behavior_correct']}")
        print(f"  Danger behavior correct: {analysis['danger_behavior_correct']}")
        print(f"  Fully trained: {analysis['fully_trained']}")
        
        return all_stats
    
    def evaluate(
        self,
        num_episodes: int = 10,
        steps_per_episode: int = 100
    ) -> Dict:
        """
        Evaluate the trained creature without learning.
        
        Returns
        -------
        dict
            Evaluation metrics
        """
        print(f"\nEvaluating for {num_episodes} episodes (no learning)...")
        
        total_reward = 0
        total_accuracy = 0
        
        for ep in range(num_episodes):
            stats = self.run_episode(
                num_steps=steps_per_episode,
                learn=False,
                verbose=False
            )
            total_reward += stats.total_reward
            total_accuracy += stats.accuracy
        
        avg_reward = total_reward / num_episodes
        avg_accuracy = total_accuracy / num_episodes
        
        print(f"Average reward: {avg_reward:.1f}")
        print(f"Average accuracy: {avg_accuracy:.1%}")
        
        return {
            "avg_reward": avg_reward,
            "avg_accuracy": avg_accuracy,
            "num_episodes": num_episodes
        }


def run_demo():
    """Run a demonstration of the R-STDP creature learning"""
    print("="*60)
    print("R-STDP Creature Learning Demo")
    print("="*60)
    print()
    
    # Create world and brain
    world = World(world_size=10, spawn_prob=0.4, despawn_prob=0.1, seed=42)
    brain = CreatureBrain(learning_rate=0.3, initial_weight=0.5, seed=123)
    
    # Create simulation
    sim = Simulation(world, brain)
    
    # Evaluate before training
    print("\n--- BEFORE TRAINING ---")
    pre_eval = sim.evaluate(num_episodes=5, steps_per_episode=50)
    
    # Train
    print("\n--- TRAINING ---")
    train_stats = sim.train(
        num_episodes=30,
        steps_per_episode=100,
        verbose_every=10
    )
    
    # Evaluate after training
    print("\n--- AFTER TRAINING ---")
    post_eval = sim.evaluate(num_episodes=5, steps_per_episode=50)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Pre-training accuracy:  {pre_eval['avg_accuracy']:.1%}")
    print(f"Post-training accuracy: {post_eval['avg_accuracy']:.1%}")
    print(f"Improvement: {(post_eval['avg_accuracy'] - pre_eval['avg_accuracy'])*100:+.1f}%")
    
    return sim


if __name__ == "__main__":
    sim = run_demo()

