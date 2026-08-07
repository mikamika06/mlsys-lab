import os
import ref

def check(workdir):
    m = {"subgraphs_count": 0.0, "nodes_accounted": 0.0}
    path = ref.create_dummy_model(workdir)
    try:
        from edge.model import get_delegation_stats
        stats = get_delegation_stats(path)
        m["subgraphs_count"] = float(stats.get("subgraphs_count", 0))
        m["nodes_accounted"] = float(stats.get("nodes_accounted", 0))
    except Exception:
        pass
    return m
