from policy.policy import PreemptionPolicy

def test_smart_policy_outperforms_pure_policies():
    policy = PreemptionPolicy(
        compute_bw=1000.0,
        pcie_bw=100000.0,
        bytes_per_tok=250.0,
        compute_ovh=2.0,
        pcie_ovh=0.1
    )

    trace = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] * 10

    smart_lats = policy.evaluate_trace(trace, "smart")
    recomp_lats = policy.evaluate_trace(trace, "recompute")
    swap_lats = policy.evaluate_trace(trace, "swap")

    smart_total = sum(smart_lats)
    recomp_total = sum(recomp_lats)
    swap_total = sum(swap_lats)

    assert smart_total < recomp_total, "Smart policy should perform strictly better than always recomputing."
    assert smart_total < swap_total, "Smart policy should perform strictly better than always swapping."
