import ref

def check(workdir):
    from routing.sim import simulate_round_robin, simulate_power_of_two
    out = {"dist_match": 0.0}
    num_replicas = 4
    reqs = [0.0, 1.0, 1.5, 2.0, 2.1, 3.0, 3.5, 4.0, 4.2, 5.0] * 10
    serv = [5.0, 1.0, 2.0, 8.0, 1.5, 3.0, 0.5, 4.0, 2.5, 1.0] * 10
    rr_c_ref, rr_l_ref = ref.simulate_round_robin(num_replicas, reqs, serv, seed=42)
    p2_c_ref, p2_l_ref = ref.simulate_power_of_two(num_replicas, reqs, serv, seed=42)
    try:
        rr_c_got, rr_l_got = simulate_round_robin(num_replicas, reqs, serv, seed=42)
        p2_c_got, p2_l_got = simulate_power_of_two(num_replicas, reqs, serv, seed=42)
    except Exception as e:
        out["_note"] = f"simulation raised error: {e}"
        return out
    if rr_c_ref == rr_c_got and p2_c_ref == p2_c_got and rr_l_ref == rr_l_got and p2_l_ref == p2_l_got:
        out["dist_match"] = 1.0
    else:
        out["_note"] = "simulation outputs do not match reference"
    return out
