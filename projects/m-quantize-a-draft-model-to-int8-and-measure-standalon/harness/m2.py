import ref


def check(workdir):
    from draft.simulate import simulate_acceptance_rates

    draft_fp16, draft_int8, target = ref.get_test_logits(n_runs=20, gamma=4, vocab_size=16)

    res = simulate_acceptance_rates(draft_fp16, draft_int8, target, gamma=4)

    want_fp16 = 0.0
    want_int8 = 0.0
    total = 20 * 4

    for i in range(20):
        for k in range(4):
            if int(draft_fp16[i, k].argmax()) == int(target[i, k].argmax()):
                want_fp16 += 1
            else:
                break

        for k in range(4):
            if int(draft_int8[i, k].argmax()) == int(target[i, k].argmax()):
                want_int8 += 1
            else:
                break

    exp_alpha_fp16 = want_fp16 / total
    exp_alpha_int8 = want_int8 / total
    exp_delta = exp_alpha_int8 - exp_alpha_fp16

    got_alpha_fp16 = float(res.get("alpha_fp16", -1.0))
    got_alpha_int8 = float(res.get("alpha_int8", -1.0))
    got_delta = float(res.get("delta", -99.0))

    rate_match = 1.0 if (abs(got_alpha_fp16 - exp_alpha_fp16) < 1e-5 and abs(got_alpha_int8 - exp_alpha_int8) < 1e-5) else 0.0
    delta_match = 1.0 if abs(got_delta - exp_delta) < 1e-5 else 0.0

    return {
        "rate_match": rate_match,
        "delta_match": delta_match
    }
