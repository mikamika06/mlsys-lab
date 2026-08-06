import numpy as np
import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from ringattn.lse import merge_lse_pair, merge_partial_outputs

    errors = []
    for seed in [12, 34, 56]:
        partials = ref.generate_lse_test_case(seed=seed)

        want_out, want_max, want_sum = ref.ref_merge_partial_outputs(partials)
        got_out, got_max, got_sum = merge_partial_outputs(partials)

        rel_out = np.max(np.abs(got_out - want_out) / (np.abs(want_out) + 1e-9))
        rel_max = np.max(np.abs(got_max - want_max) / (np.abs(want_max) + 1e-9))
        rel_sum = np.max(np.abs(got_sum - want_sum) / (np.abs(want_sum) + 1e-9))

        errors.extend([rel_out, rel_max, rel_sum])

    max_rel_err = float(np.max(errors))
    return {"rel_err": max_rel_err}
