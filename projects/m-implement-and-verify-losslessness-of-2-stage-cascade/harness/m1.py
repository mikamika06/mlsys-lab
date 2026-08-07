import numpy as np
import ref


def check(workdir):
    from cascade.sampling import cascade_stage1_accept, cascade_stage2_accept, multi_draft_select

    out = {"max_abs_err": 1.0, "lossless_matched": 0.0}
    max_err = 0.0
    matches = True

    for i, (q1, q2, p) in enumerate(ref.DISTRIBUTIONS):
        rng_ref = np.random.default_rng(100 + i)
        rng_usr = np.random.default_rng(100 + i)

        for step in range(200):
            x1 = int(rng_ref.choice(len(q1), p=q1))
            x1_usr = int(rng_usr.choice(len(q1), p=q1))

            r_acc1, r_x2 = ref.cascade_stage1_accept(q1, q2, x1, rng_ref)
            u_acc1, u_x2 = cascade_stage1_accept(q1, q2, x1_usr, rng_usr)

            if r_acc1 != u_acc1 or r_x2 != u_x2:
                matches = False
                if "_note" not in out:
                    out["_note"] = f"stage1 mismatch at dist {i} step {step}: ref=({r_acc1},{r_x2}) usr=({u_acc1},{u_x2})"

            r_acc2, r_xf = ref.cascade_stage2_accept(q2, p, r_x2, rng_ref)
            u_acc2, u_xf = cascade_stage2_accept(q2, p, u_x2, rng_usr)

            if r_acc2 != u_acc2 or r_xf != u_xf:
                matches = False
                if "_note" not in out:
                    out["_note"] = f"stage2 mismatch at dist {i} step {step}: ref=({r_acc2},{r_xf}) usr=({u_acc2},{u_xf})"

            c0 = int(rng_ref.choice(len(q1), p=q1))
            c1 = int(rng_ref.choice(len(q2), p=q2))
            c0_u = int(rng_usr.choice(len(q1), p=q1))
            c1_u = int(rng_usr.choice(len(q2), p=q2))

            r_m_acc, r_m_idx, r_m_tok = ref.multi_draft_select([c0, c1], [q1, q2], p, rng_ref)
            u_m_acc, u_m_idx, u_m_tok = multi_draft_select([c0_u, c1_u], [q1, q2], p, rng_usr)

            if r_m_acc != u_m_acc or r_m_idx != u_m_idx or r_m_tok != u_m_tok:
                matches = False
                if "_note" not in out:
                    out["_note"] = f"multi_draft mismatch at dist {i} step {step}"

        rng_stat = np.random.default_rng(2000 + i)
        num_samples = 30000
        counts = np.zeros(len(p))
        for _ in range(num_samples):
            c0 = int(rng_stat.choice(len(q1), p=q1))
            c1 = int(rng_stat.choice(len(q2), p=q2))
            _, _, x_final = multi_draft_select([c0, c1], [q1, q2], p, rng_stat)
            counts[x_final] += 1
        p_emp = counts / num_samples
        err = float(np.max(np.abs(p_emp - p)))
        if err > max_err:
            max_err = err

    out["max_abs_err"] = float(max_err)
    out["lossless_matched"] = 1.0 if matches else 0.0
    return out
