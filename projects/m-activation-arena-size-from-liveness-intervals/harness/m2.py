import ref


def check(workdir):
    from arena.planner import plan_activation_arena

    out = {"arena_plans_matched": 0.0, "optimal_packing": 0.0}
    plans_ok = 0
    packing_ok = 0
    total = len(ref.BUFFER_TEST_CASES)

    for i, bufs in enumerate(ref.BUFFER_TEST_CASES):
        want = ref.reference_plan_activation_arena(bufs)
        try:
            got = plan_activation_arena(bufs)
        except Exception as e:
            out["_note"] = f"case {i} raised exception: {e}"
            return out

        if got == want:
            plans_ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i} plan mismatch: got {got}, want {want}"

        if isinstance(got, dict) and got.get("arena_size") == want["arena_size"]:
            packing_ok += 1

    if plans_ok == total:
        out["arena_plans_matched"] = 1.0
    if packing_ok == total:
        out["optimal_packing"] = 1.0

    return out
