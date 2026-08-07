import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        import autotune
        configs = ref.generate_configs()
        rk = ["BLOCK_M", "BLOCK_N", "num_warps"]
        oom_configs = [{"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4}]

        ok = 0
        for c in configs:
            want = ref.is_dominated(c, oom_configs, rk)
            try:
                got = autotune.is_dominated(c, oom_configs, rk)
                if got == want:
                    ok += 1
            except Exception:
                pass

        out = {"is_dominated_correct": 1.0 if ok == len(configs) else 0.0}
    except Exception:
        out = {"is_dominated_correct": 0.0}
    finally:
        sys.path.pop(0)

    return out
