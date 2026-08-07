import sys

sys.path.insert(0, ".")
from roof.rank import rank_device_kernel_pairs

PAIRS = [
    {"device": "GPU_A", "flops": 200.0, "bytes": 50.0, "peak_flop": 100.0, "peak_bw": 10.0},
    {"device": "GPU_A", "flops": 1500.0, "bytes": 100.0, "peak_flop": 100.0, "peak_bw": 10.0},
    {"device": "GPU_B", "flops": 100.0, "bytes": 80.0, "peak_flop": 50.0, "peak_bw": 20.0},
]


def test_ranking_orders_by_distance_from_ridge():
    ranked = rank_device_kernel_pairs(PAIRS)
    distances = []
    for p in ranked:
        ridge = p["peak_flop"] / p["peak_bw"]
        intensity = p["flops"] / p["bytes"]
        distances.append(intensity - ridge)
    for i in range(len(distances) - 1):
        assert distances[i] <= distances[i + 1], "pairs are not sorted correctly by distance"
