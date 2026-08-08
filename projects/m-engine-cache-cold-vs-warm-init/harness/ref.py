SCENARIOS = [
    {
        "engine_meta": {"hash": "h1", "profile_signature": "p1", "plugin_version": "v1"},
        "cache_store": {},
        "expected": "cold"
    },
    {
        "engine_meta": {"hash": "h2", "profile_signature": "p1", "plugin_version": "v1"},
        "cache_store": {"h2": {"profile_signature": "p1", "plugin_version": "v1"}},
        "expected": "warm"
    },
    {
        "engine_meta": {"hash": "h3", "profile_signature": "p2", "plugin_version": "v1"},
        "cache_store": {"h3": {"profile_signature": "p1", "plugin_version": "v1"}},
        "expected": "invalidated"
    },
    {
        "engine_meta": {"hash": "h4", "profile_signature": "p1", "plugin_version": "v2"},
        "cache_store": {"h4": {"profile_signature": "p1", "plugin_version": "v1"}},
        "expected": "invalidated"
    }
]

GRAPH_NODES = ["node_0", "node_1", "node_2", "node_3", "node_4"]
SUBGRAPHS = [{"nodes": ["node_0", "node_1", "node_2"]}, {"nodes": ["node_3", "node_4"]}]
