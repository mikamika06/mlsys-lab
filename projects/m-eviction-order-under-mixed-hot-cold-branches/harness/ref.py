CONFIGS = [
    {
        "nodes": {
            0: {"parent": None, "children": [1, 2], "access_count": 100, "last_access": 10, "is_hot": True},
            1: {"parent": 0, "children": [3], "access_count": 80, "last_access": 9, "is_hot": True},
            2: {"parent": 0, "children": [], "access_count": 5, "last_access": 1, "is_hot": False},
            3: {"parent": 1, "children": [], "access_count": 70, "last_access": 8, "is_hot": True},
        },
        "capacity": 2
    },
    {
        "nodes": {
            0: {"parent": None, "children": [1], "access_count": 200, "last_access": 20, "is_hot": True},
            1: {"parent": 0, "children": [2, 3], "access_count": 150, "last_access": 15, "is_hot": True},
            2: {"parent": 1, "children": [], "access_count": 2, "last_access": 2, "is_hot": False},
            3: {"parent": 1, "children": [], "access_count": 10, "last_access": 5, "is_hot": False},
        },
        "capacity": 3
    },
    {
        "nodes": {
            0: {"parent": None, "children": [1, 2], "access_count": 50, "last_access": 5, "is_hot": False},
            1: {"parent": 0, "children": [], "access_count": 40, "last_access": 4, "is_hot": False},
            2: {"parent": 0, "children": [], "access_count": 10, "last_access": 1, "is_hot": False},
        },
        "capacity": 1
    }
]

def compute_scores(config):
    res = {}
    for nid, node in config["nodes"].items():
        base = node["access_count"] * 2 + node["last_access"]
        if node["is_hot"]:
            base *= 10
        res[nid] = float(base)
    return res

def eviction_order(config):
    scores = compute_scores(config)
    leaves = [n for n, node in config["nodes"].items() if not node["children"]]
    leaves.sort(key=lambda x: (scores[x], x))
    return leaves

def simulate_reclaim(config):
    order = eviction_order(config)
    cap = config["capacity"]
    total_nodes = len(config["nodes"])
    freed = max(0, total_nodes - cap)
    return order[:freed]

def check_safety(config):
    order = eviction_order(config)
    scores = compute_scores(config)
    for i in range(len(order) - 1):
        assert scores[order[i]] <= scores[order[i+1]]
    return True
