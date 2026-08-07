import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        import quant.scale as qs

        out = {"mse_ratio": 0.0}
        mse_opt = 0.0
        mse_max = 0.0

        for i in range(len(ref.W_TEST)):
            chan = ref.W_TEST[i]
            s_opt = qs.find_best_scale_mse(chan)
            s_max = qs.compute_max_scale(chan)

            dq_opt = qs.simulate_quant(chan, s_opt)
            dq_max = qs.simulate_quant(chan, s_max)

            mse_opt += np.mean((chan - dq_opt) ** 2)
            mse_max += np.mean((chan - dq_max) ** 2)

        if mse_opt > 0:
            out["mse_ratio"] = mse_max / mse_opt

        return out
    finally:
        sys.path.pop(0)
