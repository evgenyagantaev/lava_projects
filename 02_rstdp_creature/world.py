"""
1D World Simulation for R-STDP Creature Learning

This module implements a simple 1D world where a creature can move
left or right, seeking food and avoiding danger.
"""

import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple


class EntityType(IntEnum):
    """Types of entities in the world"""
    NONE = 0
    FOOD = 1
    DANGER = 2


class Action(IntEnum):
    """Possible actions for the creature"""
    STAY = 0
    LEFT = 1
    RIGHT = 2


@dataclass
class WorldState:
    """Current state of the world"""
    creature_pos: float
    entity_type: EntityType
    entity_pos: Optional[float]
    
    def __str__(self) -> str:
        entity_str = "None"
        if self.entity_type == EntityType.FOOD:
            entity_str = f"Food at {self.entity_pos}"
        elif self.entity_type == EntityType.DANGER:
            entity_str = f"Danger at {self.entity_pos}"
        return f"Creature at {self.creature_pos}, {entity_str}"


class World:
    """
    1D World simulation.
    
    The world is a line segment from -world_size to +world_size.
    A creature lives in this world and can move left or right.
    Food and danger can appear randomly (never simultaneously).
    
    Parameters
    ----------
    world_size : float
        Half-width of the world (world spans from -world_size to +world_size)
    spawn_prob : float
        Probability of spawning food or danger each step (when none exists)
    despawn_prob : float
        Probability of despawning food or danger each step
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        world_size: float = 10.0,
        spawn_prob: float = 0.3,
        despawn_prob: float = 0.1,
        seed: Optional[int] = None
    ):
        self.world_size = world_size
        self.spawn_prob = spawn_prob
        self.despawn_prob = despawn_prob
        
        self.rng = np.random.default_rng(seed)
        
        # State
        self.creature_pos = 0.0
        self.entity_type = EntityType.NONE
        self.entity_pos = None
        
        self.step_count = 0
    
    def reset(self) -> WorldState:
        """Reset world to initial state"""
        self.creature_pos = 0.0
        self.entity_type = EntityType.NONE
        self.entity_pos = None
        self.step_count = 0
        return self.get_state()
    
    def get_state(self) -> WorldState:
        """Get current world state"""
        return WorldState(
            creature_pos=self.creature_pos,
            entity_type=self.entity_type,
            entity_pos=self.entity_pos
        )
    
    def get_sensory_input(self) -> Tuple[int, int, int, int]:
        """
        Get sensory input for the creature.
        
        Returns
        -------
        tuple of 4 ints (food_left, food_right, danger_left, danger_right)
            Each is 1 if the entity is in that direction, 0 otherwise
        """
        food_left = 0
        food_right = 0
        danger_left = 0
        danger_right = 0
        
        if self.entity_type == EntityType.FOOD and self.entity_pos is not None:
            if self.entity_pos < self.creature_pos:
                food_left = 1
            elif self.entity_pos > self.creature_pos:
                food_right = 1
        elif self.entity_type == EntityType.DANGER and self.entity_pos is not None:
            if self.entity_pos < self.creature_pos:
                danger_left = 1
            elif self.entity_pos > self.creature_pos:
                danger_right = 1
        
        return (food_left, food_right, danger_left, danger_right)
    
    def step(self, action: Action) -> Tuple[WorldState, float, dict]:
        """
        Execute one step in the world.
        
        Parameters
        ----------
        action : Action
            The action to take (STAY, LEFT, or RIGHT)
        
        Returns
        -------
        state : WorldState
            New world state
        reward : float
            Reward signal for the action
        info : dict
            Additional information about the step
        """
        self.step_count += 1
        old_pos = self.creature_pos
        
        # Execute action
        if action == Action.LEFT:
            self.creature_pos = max(-self.world_size, self.creature_pos - 1)
        elif action == Action.RIGHT:
            self.creature_pos = min(self.world_size, self.creature_pos + 1)
        
        # Calculate reward
        reward = self._calculate_reward(old_pos, action)
        
        # Check if creature reached entity
        reached_entity = False
        if self.entity_pos is not None:
            if abs(self.creature_pos - self.entity_pos) < 0.5:
                reached_entity = True
                # Despawn entity when reached
                self.entity_type = EntityType.NONE
                self.entity_pos = None
        
        # Update entity spawning/despawning
        self._update_entities()
        
        info = {
            "step": self.step_count,
            "action": action,
            "reached_entity": reached_entity
        }
        
        return self.get_state(), reward, info
    
    def _calculate_reward(self, old_pos: float, action: Action) -> float:
        """
        Calculate reward based on action and entity positions.
        
        Reward logic:
        - +1 if moved towards food
        - +1 if moved away from danger
        - 0 otherwise
        """
        if self.entity_pos is None or action == Action.STAY:
            return 0.0
        
        # Direction of movement
        moved_left = (action == Action.LEFT)
        moved_right = (action == Action.RIGHT)
        
        # Direction to entity
        entity_is_left = (self.entity_pos < old_pos)
        entity_is_right = (self.entity_pos > old_pos)
        
        if self.entity_type == EntityType.FOOD:
            # Reward for moving towards food
            if (moved_left and entity_is_left) or (moved_right and entity_is_right):
                return 1.0
            else:
                return 0.0
        
        elif self.entity_type == EntityType.DANGER:
            # Reward for moving away from danger
            if (moved_left and entity_is_right) or (moved_right and entity_is_left):
                return 1.0
            else:
                return 0.0
        
        return 0.0
    
    def _update_entities(self):
        """Update entity spawning and despawning"""
        if self.entity_type == EntityType.NONE:
            # Try to spawn
            if self.rng.random() < self.spawn_prob:
                # Randomly choose food or danger
                self.entity_type = EntityType.FOOD if self.rng.random() < 0.5 else EntityType.DANGER
                
                # Spawn at random position (not too close to creature)
                while True:
                    pos = self.rng.uniform(-self.world_size, self.world_size)
                    if abs(pos - self.creature_pos) > 2.0:
                        self.entity_pos = pos
                        break
        else:
            # Try to despawn
            if self.rng.random() < self.despawn_prob:
                self.entity_type = EntityType.NONE
                self.entity_pos = None
    
    def render_ascii(self, width: int = 40) -> str:
        """
        Render world as ASCII art.
        
        Parameters
        ----------
        width : int
            Width of the ASCII representation
        
        Returns
        -------
        str
            ASCII representation of the world
        """
        # Map positions to character indices
        def pos_to_idx(pos):
            return int((pos + self.world_size) / (2 * self.world_size) * (width - 1))
        
        # Create empty line
        line = ['-'] * width
        
        # Place entity
        if self.entity_pos is not None:
            idx = pos_to_idx(self.entity_pos)
            if self.entity_type == EntityType.FOOD:
                line[idx] = 'F'
            elif self.entity_type == EntityType.DANGER:
                line[idx] = 'X'
        
        # Place creature (overwrites entity if at same position)
        creature_idx = pos_to_idx(self.creature_pos)
        line[creature_idx] = 'C'
        
        # Build output
        result = ''.join(line)
        legend = f"C={self.creature_pos:.1f}"
        if self.entity_type == EntityType.FOOD:
            legend += f", Food={self.entity_pos:.1f}"
        elif self.entity_type == EntityType.DANGER:
            legend += f", Danger={self.entity_pos:.1f}"
        
        return f"|{result}|\n {legend}"


if __name__ == "__main__":
    # Quick test
    world = World(seed=42)
    state = world.reset()
    
    print("Initial state:")
    print(world.render_ascii())
    print()
    
    # Run a few random steps
    for i in range(10):
        action = Action(np.random.randint(0, 3))
        state, reward, info = world.step(action)
        print(f"Step {i+1}: Action={action.name}, Reward={reward}")
        print(world.render_ascii())
        print(f"Sensors: {world.get_sensory_input()}")
        print()

