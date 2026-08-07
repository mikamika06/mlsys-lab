"""MPSGraph dispatch vs loop-of-ops analysis."""


def simulate_loop_of_ops_dispatches(ops_count):
    return {
        "command_buffers": ops_count,
        "encoded_commands": ops_count,
        "kind": "loop_of_ops",
    }


def simulate_mpsgraph_dispatches(ops_count):
    cb_count = 1 if ops_count > 0 else 0
    return {
        "command_buffers": cb_count,
        "encoded_commands": ops_count,
        "kind": "mpsgraph",
    }


def compute_consolidation_ratio(ops_count):
    if ops_count <= 0:
        return 1.0
    return float(ops_count)
