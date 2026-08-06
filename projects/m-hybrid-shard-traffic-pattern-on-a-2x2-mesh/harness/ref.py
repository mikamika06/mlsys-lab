from reference.mesh.traffic import simulate_traffic
from reference.mesh.policy import diagnose_wrap_policy
from reference.mesh.strategy import select_strategy

STRATEGIES = ["HYBRID_SHARD", "FULL_SHARD_1D", "NO_SHARD"]
CONFIGS = [
    {"min_size": 1024},
    {"min_size": 2048},
    {"min_size": 512}
]
MODULE_SETS = [
    [{"name": "layer1", "size": 2048, "wrapped": True}, {"name": "layer2", "size": 500, "wrapped": False}],
    [{"name": "layer1", "size": 4096, "wrapped": True}],
    [{"name": "layer1", "size": 100, "wrapped": False}]
]
BUDGETS = [
    (10000, 10000),
    (10000, 2500),
    (10000, 1000)
]
