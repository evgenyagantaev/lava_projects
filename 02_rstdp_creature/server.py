"""
Simple HTTP server for R-STDP Deep Creature Animation

Uses built-in Python http.server - no external dependencies!
Supports multi-layer neural network visualization.
"""

import http.server
import socketserver
import json
import os
from urllib.parse import urlparse, parse_qs

# Change to the script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from world import World, Action, EntityType
from creature import DeepCreatureBrain


class SimState:
    """Global simulation state with deep neural network"""
    
    def __init__(self):
        self.world = None
        self.brain = None
        self.running = False
        self.speed = 400  # ms
        self.total_reward = 0
        self.episode = 0
        self.step = 0
        self.correct_count = 0
        self.total_decisions = 0
        self.last_result = {}
        
    def reset(self, seed=None):
        """Reset simulation with new deep network"""
        self.world = World(
            world_size=10.0,
            spawn_prob=0.5,
            despawn_prob=0.15,
            seed=seed
        )
        
        # Use deep brain with eligibility traces!
        self.brain = DeepCreatureBrain(
            hidden_sizes=[6, 4],      # Two hidden layers
            learning_rate=0.25,
            trace_decay=0.85,         # Eligibility trace decay
            weight_decay=0.002,       # Forgetting rate
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
            "episode": self.episode,
            "step": self.step,
            "accuracy": accuracy,
            "running": self.running,
            "speed": self.speed,
            "last_result": self.last_result
        }
    
    def do_step(self):
        """Execute one simulation step"""
        if not self.world or not self.brain:
            return {"error": "Not initialized"}
        
        sensory = self.world.get_sensory_input()
        action = self.brain.decide(sensory)
        old_pos = self.world.creature_pos
        state, reward, info = self.world.step(action)
        
        correct = False
        panic = False
        food_left, food_right, danger_left, danger_right = sensory
        
        if any(sensory):
            self.total_decisions += 1
            if food_left and action == Action.LEFT:
                correct = True
            elif food_right and action == Action.RIGHT:
                correct = True
            elif danger_left and action == Action.RIGHT:
                correct = True
            elif danger_right and action == Action.LEFT:
                correct = True
            else:
                panic = True
            
            if correct:
                self.correct_count += 1
        
        # Learn with reward
        if reward != 0:
            self.brain.learn(sensory, action, reward)
        else:
            # Still update eligibility traces (decay happens)
            self.brain.learn(sensory, action, 0)
        
        self.total_reward += reward
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
        elif parsed.path == "/" or parsed.path == "/index.html":
            self.path = "/index.html"
            super().do_GET()
        else:
            super().do_GET()
    
    def send_json(self, data):
        content = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(content))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)
    
    def log_message(self, format, *args):
        if "/api/" not in args[0]:
            super().log_message(format, *args)


def run_server(port=8766):
    with socketserver.TCPServer(("", port), RequestHandler) as httpd:
        print(f"\n{'='*60}")
        print("  R-STDP Deep Creature Animation Server")
        print(f"{'='*60}")
        print(f"\n  Architecture: 4 → 6 → 4 → 2 (56 plastic synapses)")
        print(f"  Features: Eligibility traces + Weight decay")
        print(f"\n  Open in browser: http://localhost:{port}")
        print("\n  Press Ctrl+C to stop\n")
        httpd.serve_forever()


if __name__ == "__main__":
    run_server()
