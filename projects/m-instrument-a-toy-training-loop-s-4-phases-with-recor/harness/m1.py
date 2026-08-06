import ref


def check(workdir):
    from profiler.loop import compute_uncovered_time_pct, profile_training_loop

    got_pcts = profile_training_loop(ref.synthetic_step_fn, num_steps=3)
    got_uncovered = compute_uncovered_time_pct(
        ref.synthetic_step_fn, num_steps=3
    )

    phases = ["forward", "loss", "backward", "optimizer"]
    has_all_phases = all(p in got_pcts for p in phases)
    sum_near_100 = abs(sum(got_pcts.values()) - 100.0) < 1e-3
    valid_uncovered = 0.0 <= got_uncovered <= 100.0

    if not (has_all_phases and sum_near_100 and valid_uncovered):
        return {
            "rel_err": 1.0,
            "_note": f"Invalid distribution or uncovered range: pcts={got_pcts}, uncovered={got_uncovered}",
        }

    err = 0.0
    if got_pcts["backward"] <= got_pcts["loss"]:
        err = 0.1

    return {"rel_err": err}
