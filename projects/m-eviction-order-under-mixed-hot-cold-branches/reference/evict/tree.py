from evict.policy import compute_scores


def eviction_order(config):
    scores = compute_scores(config)
    leaves = [n for n, node in config["nodes"].items() if not node["children"]]
    leaves.sort(key=lambda x: (scores[x], x))
    return leaves
