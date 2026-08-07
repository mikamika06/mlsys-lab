import numpy as np
from striped.policy import assign_blocks
from striped.simulator import simulate_throughput

SCENARIOS = [
    {"num_blocks": 16, "world_size": 4, "compute_cost": 10.0, "comm_cost": 2.0},
    {"num_blocks": 32, "world_size": 8, "compute_cost": 5.0, "comm_cost": 1.0},
    {"num_blocks": 8, "world_size": 2, "compute_cost": 20.0, "comm_cost": 4.0},
]
