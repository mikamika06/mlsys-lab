import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from triton_remapping.remapping import remap_pid, generate_grid_schedule
    except Exception as e:
        return {"remapping_matched": 0.0, "_note": f"Import error: {e}"}

    matched = True
    for cfg in ref.TEST_CONFIGS:
        m, n, g = cfg["num_pid_m"], cfg["num_pid_n"], cfg["group_size_m"]
        total = m * n
        for pid in range(total):
            want = ref.remap_pid(pid, m, n, g)
            got = remap_pid(pid, m, n, g)
            if got != want:
                matched = False
                break
        if not matched:
            break

        want_sched = ref.generate_grid_schedule(m, n, g)
        got_sched = generate_grid_schedule(m, n, g)
        if got_sched != want_sched:
            matched = False
            break

    return {"remapping_matched": 1.0 if matched else 0.0}
