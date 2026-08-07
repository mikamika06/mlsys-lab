def analyze_nixl_faults(topology, failure_mode):
    nodes = topology.get("nodes", [])
    if failure_mode == "server_crash":
        return {
            "affected_nodes": [n["id"] for n in nodes],
            "blast_radius": "cluster_wide",
            "mitigation": "enable_raft_ha"
        }
    elif failure_mode == "heartbeat_timeout":
        return {
            "affected_nodes": [n["id"] for n in nodes if n.get("role") == "decode"],
            "blast_radius": "decode_pool",
            "mitigation": "adjust_timeout_threshold"
        }
    elif failure_mode == "stale_metadata":
        return {
            "affected_nodes": [n["id"] for n in nodes if n.get("tier") == "edge"],
            "blast_radius": "isolated_node",
            "mitigation": "force_refresh"
        }
    return {
        "affected_nodes": [],
        "blast_radius": "none",
        "mitigation": "none"
    }
