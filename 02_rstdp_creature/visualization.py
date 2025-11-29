"""
Visualization utilities for R-STDP Creature Learning

Provides functions to visualize:
- Training progress (reward, accuracy over episodes)
- Weight evolution
- Creature behavior
- World state
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation
from typing import List, Optional

from simulation import Simulation, EpisodeStats


def plot_training_progress(
    episode_stats: List[EpisodeStats],
    title: str = "Training Progress",
    figsize: tuple = (12, 8)
) -> plt.Figure:
    """
    Plot training metrics over episodes.
    
    Parameters
    ----------
    episode_stats : list of EpisodeStats
        Statistics from training episodes
    title : str
        Plot title
    figsize : tuple
        Figure size
    
    Returns
    -------
    Figure
        Matplotlib figure
    """
    episodes = range(1, len(episode_stats) + 1)
    rewards = [s.total_reward for s in episode_stats]
    accuracies = [s.accuracy * 100 for s in episode_stats]
    decisions = [s.total_decisions for s in episode_stats]
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Reward plot
    ax = axes[0]
    ax.plot(episodes, rewards, 'b-', linewidth=2, label='Total Reward')
    ax.axhline(y=np.mean(rewards), color='b', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(rewards):.1f}')
    ax.set_ylabel('Total Reward')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    
    # Accuracy plot
    ax = axes[1]
    ax.plot(episodes, accuracies, 'g-', linewidth=2, label='Accuracy')
    ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Random (50%)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Decisions plot
    ax = axes[2]
    ax.bar(episodes, decisions, color='orange', alpha=0.7, label='Decisions with entity')
    ax.set_xlabel('Episode')
    ax.set_ylabel('# Decisions')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_weight_evolution(
    weight_history: List[np.ndarray],
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Plot evolution of synaptic weights over time.
    
    Parameters
    ----------
    weight_history : list of np.ndarray
        List of weight matrices over time
    figsize : tuple
        Figure size
    
    Returns
    -------
    Figure
        Matplotlib figure
    """
    if len(weight_history) == 0:
        return None
    
    # Stack weights: (time, num_motor, num_sensory)
    weights = np.array(weight_history)
    time = np.arange(len(weights))
    
    num_motor, num_sensory = weights.shape[1], weights.shape[2]
    
    sensory_names = ['Food Left', 'Food Right', 'Danger Left', 'Danger Right']
    motor_names = ['Move Left', 'Move Right']
    
    fig, axes = plt.subplots(num_motor, 1, figsize=figsize, sharex=True)
    if num_motor == 1:
        axes = [axes]
    
    colors = ['green', 'lime', 'red', 'orange']
    
    for motor_idx, ax in enumerate(axes):
        for sensory_idx in range(num_sensory):
            ax.plot(
                time,
                weights[:, motor_idx, sensory_idx],
                color=colors[sensory_idx],
                linewidth=2,
                label=sensory_names[sensory_idx]
            )
        
        ax.set_ylabel(f'{motor_names[motor_idx]}\nWeight')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Learning Step')
    axes[0].set_title('Synaptic Weight Evolution')
    
    plt.tight_layout()
    return fig


def plot_weight_matrix(
    weights: np.ndarray,
    title: str = "Weight Matrix",
    figsize: tuple = (8, 4)
) -> plt.Figure:
    """
    Plot current weight matrix as heatmap.
    
    Parameters
    ----------
    weights : np.ndarray
        Weight matrix (num_motor, num_sensory)
    title : str
        Plot title
    figsize : tuple
        Figure size
    
    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sensory_names = ['Food\nLeft', 'Food\nRight', 'Danger\nLeft', 'Danger\nRight']
    motor_names = ['Move Left', 'Move Right']
    
    im = ax.imshow(weights, cmap='RdYlGn', aspect='auto', vmin=0, vmax=np.max(weights) * 1.2)
    
    # Labels
    ax.set_xticks(range(len(sensory_names)))
    ax.set_xticklabels(sensory_names)
    ax.set_yticks(range(len(motor_names)))
    ax.set_yticklabels(motor_names)
    
    # Add text annotations
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            text = ax.text(j, i, f'{weights[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=12)
    
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Weight')
    
    plt.tight_layout()
    return fig


def plot_world_state(
    creature_pos: float,
    entity_type: int,
    entity_pos: Optional[float],
    world_size: float = 10.0,
    figsize: tuple = (12, 3)
) -> plt.Figure:
    """
    Visualize current world state.
    
    Parameters
    ----------
    creature_pos : float
        Creature's position
    entity_type : int
        0=None, 1=Food, 2=Danger
    entity_pos : float or None
        Entity position
    world_size : float
        World size
    figsize : tuple
        Figure size
    
    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw world line
    ax.axhline(y=0, color='gray', linewidth=2)
    ax.axvline(x=-world_size, color='gray', linewidth=1, linestyle='--')
    ax.axvline(x=world_size, color='gray', linewidth=1, linestyle='--')
    
    # Draw creature
    creature_circle = Circle((creature_pos, 0), 0.5, color='blue', zorder=10)
    ax.add_patch(creature_circle)
    ax.annotate('Creature', (creature_pos, 0.8), ha='center', fontsize=10, color='blue')
    
    # Draw entity
    if entity_pos is not None:
        if entity_type == 1:  # Food
            color = 'green'
            marker = 's'
            label = 'Food'
        else:  # Danger
            color = 'red'
            marker = 'X'
            label = 'Danger'
        
        ax.plot(entity_pos, 0, marker=marker, markersize=20, color=color, zorder=5)
        ax.annotate(label, (entity_pos, 0.8), ha='center', fontsize=10, color=color)
    
    # Setup axes
    ax.set_xlim(-world_size - 1, world_size + 1)
    ax.set_ylim(-1, 2)
    ax.set_xlabel('Position')
    ax.set_yticks([])
    ax.set_title('1D World')
    
    # Add position labels
    ax.set_xticks(range(int(-world_size), int(world_size) + 1, 2))
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def plot_behavior_comparison(
    pre_stats: List[EpisodeStats],
    post_stats: List[EpisodeStats],
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Compare behavior before and after training.
    
    Parameters
    ----------
    pre_stats : list of EpisodeStats
        Statistics before training
    post_stats : list of EpisodeStats
        Statistics after training
    figsize : tuple
        Figure size
    
    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Pre-training
    pre_correct = sum(s.correct_decisions for s in pre_stats)
    pre_total = sum(s.total_decisions for s in pre_stats)
    pre_wrong = pre_total - pre_correct
    
    ax = axes[0]
    ax.pie([pre_correct, pre_wrong], labels=['Correct', 'Wrong'],
           colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
    ax.set_title('Before Training')
    
    # Post-training
    post_correct = sum(s.correct_decisions for s in post_stats)
    post_total = sum(s.total_decisions for s in post_stats)
    post_wrong = post_total - post_correct
    
    ax = axes[1]
    ax.pie([post_correct, post_wrong], labels=['Correct', 'Wrong'],
           colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
    ax.set_title('After Training')
    
    fig.suptitle('Behavior Comparison: Before vs After Training', fontsize=14)
    plt.tight_layout()
    return fig


def create_animation(
    sim: Simulation,
    num_steps: int = 50,
    interval: int = 200,
    figsize: tuple = (12, 4)
):
    """
    Create animation of creature behavior.
    
    Parameters
    ----------
    sim : Simulation
        Simulation object
    num_steps : int
        Number of steps to animate
    interval : int
        Milliseconds between frames
    figsize : tuple
        Figure size
    
    Returns
    -------
    FuncAnimation
        Matplotlib animation object
    """
    sim.reset()
    
    fig, ax = plt.subplots(figsize=figsize)
    world_size = sim.world.world_size
    
    # Initial setup
    ax.set_xlim(-world_size - 1, world_size + 1)
    ax.set_ylim(-1, 2)
    ax.axhline(y=0, color='gray', linewidth=2)
    
    creature_dot, = ax.plot([], [], 'bo', markersize=20)
    entity_dot, = ax.plot([], [], 'gs', markersize=15)
    info_text = ax.text(0, 1.5, '', ha='center', fontsize=12)
    
    def init():
        creature_dot.set_data([], [])
        entity_dot.set_data([], [])
        info_text.set_text('')
        return creature_dot, entity_dot, info_text
    
    def update(frame):
        # Step simulation
        step_info = sim.step(learn=False, verbose=False)
        
        # Update creature
        creature_dot.set_data([step_info['creature_pos']], [0])
        
        # Update entity
        if step_info['entity_pos'] is not None:
            entity_dot.set_data([step_info['entity_pos']], [0])
            if step_info['entity_type'] == 1:
                entity_dot.set_color('green')
                entity_dot.set_marker('s')
            else:
                entity_dot.set_color('red')
                entity_dot.set_marker('X')
        else:
            entity_dot.set_data([], [])
        
        # Update info
        sensory = step_info['sensory']
        action = step_info['action']
        reward = step_info['reward']
        info_text.set_text(f'Step {frame+1}: Action={action.name}, Reward={reward:+.0f}')
        
        return creature_dot, entity_dot, info_text
    
    anim = FuncAnimation(
        fig, update, frames=num_steps,
        init_func=init, blit=True, interval=interval
    )
    
    return anim


if __name__ == "__main__":
    # Test visualization
    from world import World
    from creature import CreatureBrain
    from simulation import Simulation
    
    # Create and train
    world = World(seed=42)
    brain = CreatureBrain(seed=123)
    sim = Simulation(world, brain)
    
    # Train a bit
    stats = sim.train(num_episodes=20, steps_per_episode=50, verbose_every=0)
    
    # Plot training progress
    fig1 = plot_training_progress(stats)
    fig1.savefig('training_progress.png', dpi=150)
    
    # Plot weight evolution
    fig2 = plot_weight_evolution(brain.weight_history)
    fig2.savefig('weight_evolution.png', dpi=150)
    
    # Plot final weights
    fig3 = plot_weight_matrix(brain.get_weights(), "Final Weights")
    fig3.savefig('final_weights.png', dpi=150)
    
    print("Visualization saved!")
    plt.show()

