def is_dominated(candidate, population):
    """Check if candidate is strictly worse or equal in all metrics and strictly worse in at least one."""
    c_lat, c_mem, c_reg = candidate["latency_us"], candidate["memory_bytes"], candidate["num_registers"]
    for p in population:
        p_lat, p_mem, p_reg = p["latency_us"], p["memory_bytes"], p["num_registers"]
        if (p_lat <= c_lat and p_mem <= c_mem and p_reg <= c_reg) and (p_lat < c_lat or p_mem < c_mem or p_reg < c_reg):
            return True
    return False


def find_pareto_frontier(records):
    """Return list of non-dominated records preserving relative order."""
    valid = [r for r in records if r.get("status") == "OK"]
    frontier = []
    for r in valid:
        if not is_dominated(r, valid):
            frontier.append(r)
    return frontier
