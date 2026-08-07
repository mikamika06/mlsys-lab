def build_execution_schedule(graph_ir):
    """Generate a topologically sorted list of node names for execution."""
    raise NotImplementedError


def verify_schedule(graph_ir, schedule):
    """Verify that every node appears after all its dependencies in the schedule."""
    raise NotImplementedError
