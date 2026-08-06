import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from profile_analyzer.utilization import compare_gpu_utilization
    from reference.profile_analyzer.utilization import compare_gpu_utilization as ref_compare

    out = {"utilization_matched": 0.0, "argmin_matched": 0.0}
    ok_util = 0
    ok_argmin = 0
    trials = 10

    for seed in range(trials):
        rep_a, rep_b = ref.generate_profile_pair(seed)
        want = ref_compare(rep_a, rep_b)
        got = compare_gpu_utilization(rep_a, rep_b)

        if (
            isinstance(got, dict)
            and abs(got.get("utilization_a", -1) - want["utilization_a"]) < 1e-5
            and abs(got.get("utilization_b", -1) - want["utilization_b"]) < 1e-5
        ):
            ok_util += 1

        if isinstance(got, dict) and got.get("argmin_index") == want["argmin_index"]:
            ok_argmin += 1

    out["utilization_matched"] = 1.0 if ok_util == trials else 0.0
    out["argmin_matched"] = 1.0 if ok_argmin == trials else 0.0
    return out
