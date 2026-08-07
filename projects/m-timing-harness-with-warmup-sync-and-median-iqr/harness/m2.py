import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        import benchmark

        out = {"matches_reference": 0.0, "handles_zero_variance": 0.0}

        samples = [1.2, 1.3, 1.1, 1.4, 1.25, 1.15, 1.35]
        want = ref.compute_required_reps(samples, 0.05, 1.96)
        try:
            got = benchmark.compute_required_reps(samples, 0.05, 1.96)
            if got == want:
                out["matches_reference"] = 1.0
        except Exception:
            pass

        samples_zero = [2.0, 2.0, 2.0, 2.0]
        want_zero = ref.compute_required_reps(samples_zero, 0.05, 1.96)
        try:
            got_zero = benchmark.compute_required_reps(samples_zero, 0.05, 1.96)
            if got_zero == want_zero:
                out["handles_zero_variance"] = 1.0
        except Exception:
            pass

        return out
    finally:
        sys.path.pop(0)
