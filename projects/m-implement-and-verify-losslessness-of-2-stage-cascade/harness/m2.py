import numpy as np
import ref


def check(workdir):
    from cascade.latency import cascade_latency_per_token, is_2stage_net_win, break_even_alpha2, expected_tokens

    out = {"latency_err": 1.0, "decisions_matched": 0.0}
    max_err = 0.0
    decisions_ok = True

    for i, cfg in enumerate(ref.LATENCY_CONFIGS):
        c1, gamma1, c2, gamma2, cT, alpha2, alpha_direct = cfg

        r_exp = ref.expected_tokens(alpha2, gamma2)
        u_exp = expected_tokens(alpha2, gamma2)
        max_err = max(max_err, abs(r_exp - u_exp))

        r_lat = ref.cascade_latency_per_token(c1, gamma1, c2, gamma2, cT, alpha2)
        u_lat = cascade_latency_per_token(c1, gamma1, c2, gamma2, cT, alpha2)
        max_err = max(max_err, abs(r_lat - u_lat))

        r_win = ref.is_2stage_net_win(c1, gamma1, c2, gamma2, cT, alpha2, alpha_direct)
        u_win = is_2stage_net_win(c1, gamma1, c2, gamma2, cT, alpha2, alpha_direct)
        if r_win != u_win:
            decisions_ok = False

        r_be = ref.break_even_alpha2(c1, gamma1, c2, gamma2, cT, alpha_direct)
        u_be = break_even_alpha2(c1, gamma1, c2, gamma2, cT, alpha_direct)
        max_err = max(max_err, abs(r_be - u_be))

    out["latency_err"] = float(max_err)
    out["decisions_matched"] = 1.0 if decisions_ok else 0.0
    return out
