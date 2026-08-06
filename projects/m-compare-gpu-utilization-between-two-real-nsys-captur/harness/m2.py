import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from profile_analyzer.churn import compute_allocation_churn
    from reference.profile_analyzer.churn import compute_allocation_churn as ref_churn

    out = {"churn_matched": 0.0}
    ok_churn = 0
    trials = 10

    for seed in range(trials):
        rep = ref.generate_churn_report(seed)
        want = ref_churn(rep)
        got = compute_allocation_churn(rep)

        if abs(got - want) < 1e-5:
            ok_churn += 1

    out["churn_matched"] = 1.0 if ok_churn == trials else 0.0
    return out
