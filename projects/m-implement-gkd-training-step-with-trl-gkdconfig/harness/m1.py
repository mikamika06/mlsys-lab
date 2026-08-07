import ref


def check(workdir):
    from gkdstep.config import GKDConfig
    from gkdstep.loss import compute_divergence, compute_gkd_step_loss

    out = {"rel_err": 1.0}
    max_err = 0.0
    modes = ["forward_kl", "reverse_kl", "jsd"]
    temperatures = [0.5, 1.0, 2.0]

    for t_logits, s_logits in ref.TEST_CASES:
        for mode in modes:
            for temp in temperatures:
                cfg = GKDConfig(temperature=temp, divergence_type=mode)
                want = ref.compute_gkd_step_loss(t_logits, s_logits, cfg)
                try:
                    got = compute_gkd_step_loss(t_logits, s_logits, cfg)
                except Exception as e:
                    out["_note"] = f"Error in compute_gkd_step_loss: {type(e).__name__}: {e}"
                    return out

                err = abs(want - got) / (abs(want) + 1e-12)
                if err > max_err:
                    max_err = err

                want_div = ref.compute_divergence(t_logits, s_logits, divergence_type=mode, temperature=temp)
                try:
                    got_div = compute_divergence(t_logits, s_logits, divergence_type=mode, temperature=temp)
                except Exception as e:
                    out["_note"] = f"Error in compute_divergence: {type(e).__name__}: {e}"
                    return out

                err_div = abs(want_div - got_div) / (abs(want_div) + 1e-12)
                if err_div > max_err:
                    max_err = err_div

    out["rel_err"] = float(max_err)
    return out
