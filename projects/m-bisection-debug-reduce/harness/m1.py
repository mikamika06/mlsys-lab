import sys
import math

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from polyreduce.bisect import bisect_divergent_step
    except Exception as e:
        return {"bisect_correct": 0.0, "is_logarithmic": 0.0, "_note": f"Import error: {e}"}

    import ref

    cases = ref.generate_bisect_cases()
    all_correct = True
    all_logarithmic = True

    for num_steps, fail_idx in cases:
        call_count = 0

        def check_fn(idx):
            nonlocal call_count
            call_count += 1
            if fail_idx == -1:
                return True
            return idx < fail_idx

        got = bisect_divergent_step(num_steps, check_fn)
        if got != fail_idx:
            all_correct = False

        max_allowed = math.ceil(math.log2(max(num_steps, 1))) + 2
        if call_count > max_allowed:
            all_logarithmic = False

    return {
        "bisect_correct": 1.0 if all_correct else 0.0,
        "is_logarithmic": 1.0 if all_logarithmic else 0.0
    }
