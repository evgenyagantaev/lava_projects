"""
Simple HTTP server for R-STDP Deep Creature Animation

Uses built-in Python http.server - no external dependencies!
Supports multi-layer neural network visualization.
Features auto-save/load state persistence.
"""

import http.server
import socketserver
import json
import os
import pickle
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Change to the script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from world import World, Action, EntityType
from creature import DeepCreatureBrain

# State file path
STATE_FILE = Path(__file__).parent / "model_state.pkl"


class SimState:
    """Global simulation state with deep neural network"""
    
    def __init__(self):
        self.world = None
        self.brain = None
        self.running = False  # Always start paused
        self.speed = 400  # ms
        self.total_reward = 0
        self.last_reward = 0  # Current reward signal
        self.episode = 0
        self.step = 0
        self.correct_count = 0
        self.total_decisions = 0
        self.last_result = {}
    
    def save_state(self):
        """Save current state to file"""
        if not self.world or not self.brain:
            return False
        
        state_data = {
            "world": {
                "creature_pos": self.world.creature_pos,
                "entity_type": self.world.entity_type.value if self.world.entity_type else None,
                "entity_pos": self.world.entity_pos,
                "step_count": self.world.step_count,
            },
            "brain": {
                "layers": [(l.weights.tolist(), l.eligibility.tolist()) for l in self.brain.layers],
                "epsilon": self.brain.epsilon,
            },
            "stats": {
                "total_reward": self.total_reward,
                "episode": self.episode,
                "step": self.step,
                "correct_count": self.correct_count,
                "total_decisions": self.total_decisions,
                "speed": self.speed,
            }
        }
        
        try:
            with open(STATE_FILE, 'wb') as f:
                pickle.dump(state_data, f)
            return True
        except Exception as e:
            print(f"Save error: {e}")
            return False
    
    def load_state(self):
        """Load state from file"""
        if not STATE_FILE.exists():
            return False
        
        try:
            with open(STATE_FILE, 'rb') as f:
                state_data = pickle.load(f)
            
            # Initialize world and brain first
            self.world = World(
                world_size=10.0,
                spawn_prob=0.6,
                despawn_prob=0.1,
            )
            self.brain = DeepCreatureBrain(
                hidden_sizes=[6, 4],
                learning_rate=5.0,
                trace_decay=0.7,
                weight_decay=0.0002,
                initial_weight=0.5,
                epsilon_start=0.3,
                epsilon_end=0.02,
                epsilon_decay=0.998,
            )
            
            # Restore world state
            self.world.creature_pos = state_data["world"]["creature_pos"]
            if state_data["world"]["entity_type"] is not None:
                self.world.entity_type = EntityType(state_data["world"]["entity_type"])
            else:
                self.world.entity_type = EntityType.NONE
            self.world.entity_pos = state_data["world"]["entity_pos"]
            self.world.step_count = state_data["world"]["step_count"]
            
            # Restore brain state
            import numpy as np
            for i, (weights, eligibility) in enumerate(state_data["brain"]["layers"]):
                self.brain.layers[i].weights = np.array(weights)
                self.brain.layers[i].eligibility = np.array(eligibility)
            self.brain.epsilon = state_data["brain"]["epsilon"]
            
            # Restore stats
            self.total_reward = state_data["stats"]["total_reward"]
            self.episode = state_data["stats"]["episode"]
            self.step = state_data["stats"]["step"]
            self.correct_count = state_data["stats"]["correct_count"]
            self.total_decisions = state_data["stats"]["total_decisions"]
            self.speed = state_data["stats"]["speed"]
            
            # Always start paused after load
            self.running = False
            self.last_result = {}
            self.last_reward = 0
            
            print(f"Loaded state: step={self.step}, accuracy={self.correct_count}/{self.total_decisions}")
            return True
            
        except Exception as e:
            print(f"Load error: {e}")
            return False
        
    def reset(self, seed=None):
        """Reset simulation with new deep network"""
        self.world = World(
            world_size=10.0,
            spawn_prob=0.6,    # Increased for more frequent entities
            despawn_prob=0.1,  # Slightly lower for longer exposure
            seed=seed
        )
        
        # Use deep brain with eligibility traces and exploration annealing!
        self.brain = DeepCreatureBrain(
            hidden_sizes=[6, 4],      # Two hidden layers
            learning_rate=5.0,        # Fast learning
            trace_decay=0.7,          # Sharper eligibility traces
            weight_decay=0.0002,      # Reduced forgetting
            initial_weight=0.5,       # Consistent initialization
            epsilon_start=0.3,        # 30% random at start
            epsilon_end=0.02,         # 2% random at end
            epsilon_decay=0.998,      # Slow decay for gradual annealing
            seed=seed
        )
        
        self.world.reset()
        self.total_reward = 0
        self.episode += 1
        self.step = 0
        self.correct_count = 0
        self.total_decisions = 0
        self.last_result = {}
        
    def to_dict(self):
        """Get current state as dictionary for JSON"""
        if not self.world or not self.brain:
            return {"error": "Simulation not initialized"}
        
        sensory = self.world.get_sensory_input()
        entity_type_str = "none"
        if self.world.entity_type == EntityType.FOOD:
            entity_type_str = "food"
        elif self.world.entity_type == EntityType.DANGER:
            entity_type_str = "danger"
        
        accuracy = 0
        if self.total_decisions > 0:
            accuracy = self.correct_count / self.total_decisions
        
        # Get deep network info
        network_stats = self.brain.get_network_stats()
        layer_info = self.brain.get_layer_info()
        
        # Format layers for JSON
        layers_data = []
        for info in layer_info:
            layers_data.append({
                "index": info["index"],
                "shape": f"{info['input_size']}→{info['output_size']}",
                "num_synapses": info["num_synapses"],
                "weights": info["weights"].tolist(),
                "eligibility": info["eligibility"].tolist(),
                "mean_weight": float(info["mean_weight"]),
                "max_weight": float(info["max_weight"]),
                "mean_eligibility": float(info["mean_eligibility"])
            })
        
        return {
            "creature_pos": self.world.creature_pos,
            "entity_type": entity_type_str,
            "entity_pos": self.world.entity_pos,
            "world_size": self.world.world_size,
            "sensory": {
                "food_left": sensory[0],
                "food_right": sensory[1],
                "danger_left": sensory[2],
                "danger_right": sensory[3]
            },
            # Deep network info
            "network": {
                "architecture": network_stats["architecture"],
                "total_synapses": network_stats["total_synapses"],
                "num_layers": network_stats["num_layers"],
                "layers": layers_data
            },
            # Backward compatible: output layer weights
            "weights": self.brain.get_weights().tolist(),
            "total_reward": self.total_reward,
            "last_reward": self.last_reward,  # Current reward signal
            "episode": self.episode,
            "step": self.step,
            "accuracy": accuracy,
            "running": self.running,
            "speed": self.speed,
            "epsilon": self.brain.epsilon if self.brain else 0,  # Exploration rate
            "last_result": self.last_result
        }
    
    def do_step(self):
        """Execute one simulation step"""
        if not self.world or not self.brain:
            return {"error": "Not initialized"}
        
        sensory = self.world.get_sensory_input()
        action = self.brain.decide(sensory)
        old_pos = self.world.creature_pos
        entity_pos = self.world.entity_pos
        state, reward, info = self.world.step(action)
        
        correct = False
        panic = False
        food_left, food_right, danger_left, danger_right = sensory
        
        if any(sensory):
            self.total_decisions += 1
            
            # Calculate distance to entity (if exists)
            distance = abs(old_pos - entity_pos) if entity_pos is not None else 999
            
            # Correct actions:
            if food_left and action == Action.LEFT:
                correct = True
            elif food_right and action == Action.RIGHT:
                correct = True
            elif danger_left and action == Action.RIGHT:
                correct = True
            elif danger_right and action == Action.LEFT:
                correct = True
            # STAY is correct when:
            elif action == Action.STAY:
                if (food_left or food_right) and distance < 1.0:
                    # Already at food - staying is correct
                    correct = True
                elif (danger_left or danger_right) and distance > 2.0:
                    # Far from danger - staying is correct
                    correct = True
                else:
                    # Should have moved
                    panic = True
            else:
                panic = True
            
            if correct:
                self.correct_count += 1
        
        # Learn with reward
        self.brain.learn(sensory, action, reward)
        
        self.total_reward += reward
        self.last_reward = reward  # Store current reward signal
        self.step += 1
        
        # Get eligibility info for visualization
        total_eligibility = sum(
            abs(layer.eligibility).sum() 
            for layer in self.brain.layers
        )
        
        self.last_result = {
            "action": action.name,
            "reward": reward,
            "correct": correct,
            "panic": panic,
            "eligibility_total": float(total_eligibility)
        }
        
        # Auto-save every 100 steps
        if self.step % 100 == 0:
            self.save_state()
        
        return self.last_result


sim_state = SimState()


class RequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with REST API"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="static", **kwargs)
    
    def do_GET(self):
        parsed = urlparse(self.path)
        
        if parsed.path == "/api/state":
            self.send_json(sim_state.to_dict())
        elif parsed.path == "/api/step":
            result = sim_state.do_step()
            self.send_json({"result": result, "state": sim_state.to_dict()})
        elif parsed.path == "/api/reset":
            params = parse_qs(parsed.query)
            seed = int(params.get("seed", [None])[0]) if params.get("seed") else None
            sim_state.reset(seed)
            self.send_json({"message": "Reset", "state": sim_state.to_dict()})
        elif parsed.path == "/api/start":
            if not sim_state.world:
                sim_state.reset()
            sim_state.running = True
            self.send_json({"message": "Started", "state": sim_state.to_dict()})
        elif parsed.path == "/api/stop":
            sim_state.running = False
            self.send_json({"message": "Stopped", "state": sim_state.to_dict()})
        elif parsed.path == "/api/speed":
            params = parse_qs(parsed.query)
            sim_state.speed = int(params.get("value", [400])[0])
            self.send_json({"message": f"Speed={sim_state.speed}", "state": sim_state.to_dict()})
        elif parsed.path == "/api/save":
            success = sim_state.save_state()
            self.send_json({"message": "Saved" if success else "Save failed", "success": success})
        elif parsed.path == "/api/load":
            success = sim_state.load_state()
            self.send_json({"message": "Loaded" if success else "No saved state", "success": success, "state": sim_state.to_dict() if success else None})
        elif parsed.path == "/favicon.ico":
            # Return empty response for favicon
            self.send_response(204)
            self.end_headers()
        elif parsed.path == "/" or parsed.path == "/index.html":
            self.path = "/index.html"
            super().do_GET()
        else:
            # Handle unknown paths gracefully
            try:
                super().do_GET()
            except Exception:
                self.send_response(404)
                self.end_headers()
    
    def send_json(self, data):
        content = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(content))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)
    
    def log_message(self, format, *args):
        # Suppress API call logs, but handle errors gracefully
        try:
            if args and isinstance(args[0], str) and "/api/" not in args[0]:
                super().log_message(format, *args)
        except:
            pass  # Ignore logging errors


def run_server(port=8766):
    # Try to load saved state on startup
    if STATE_FILE.exists():
        print("\n  Loading saved state...")
        if sim_state.load_state():
            print(f"  Restored: step={sim_state.step}, accuracy={sim_state.correct_count}/{sim_state.total_decisions}")
        else:
            print("  Failed to load, will start fresh")
            sim_state.reset()
    else:
        print("\n  No saved state found, initializing fresh...")
        sim_state.reset()
    
    # Always start paused - user controls from frontend
    sim_state.running = False
    
    with socketserver.TCPServer(("", port), RequestHandler) as httpd:
        print(f"\n{'='*60}")
        print("  R-STDP Deep Creature Animation Server")
        print(f"{'='*60}")
        print(f"\n  Architecture: 4 → 6 → 4 → 2 (56 plastic synapses)")
        print(f"  Features: Eligibility traces + Weight decay + STAY action")
        print(f"  Auto-save: every 100 steps to model_state.pkl")
        print(f"\n  Open in browser: http://localhost:{port}")
        print("\n  Press Ctrl+C to stop\n")
        httpd.serve_forever()


if __name__ == "__main__":
    run_server()
