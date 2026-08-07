import ref


def check(workdir):
    from striped.simulator import simulate_throughput

    out = {"ratios_matched": 0.0}
    ok = 0
    for i, scn in enumerate(ref.SCENARIOS):
        nb = scn["num_blocks"]
        ws = scn["world_size"]
        cc = scn["compute_cost"]
        mc = scn["comm_cost"]
        want = ref.simulate_throughput(nb, ws, cc, mc)
        got = simulate_throughput(nb, ws, cc, mc)
        if abs(got - want) < 1e-5:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"scenario {i}: got {got}, reference {want}"
    if ok == len(ref.SCENARIOS):
        out["ratios_matched"] = 1.0
    return out
