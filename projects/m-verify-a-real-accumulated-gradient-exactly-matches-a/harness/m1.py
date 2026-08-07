import ref
import numpy as np


def check(workdir):
    from gradacc.accumulate import accumulate, full_batch

    W = np.random.randn(50, 50)
    b = np.zeros(50)
    batches = ref.get_fixed_batches(4)

    dW_acc, db_acc = accumulate(batches, W, b, 4)

    X_full = np.concatenate([batch[0] for batch in batches], axis=0)
    Y_full = np.concatenate([batch[1] for batch in batches], axis=0)
    dW_full, db_full = full_batch(X_full, Y_full, W, b)

    err_w = np.max(np.abs(dW_acc - dW_full))
    err_b = np.max(np.abs(db_acc - db_full))

    return {"max_abs_err": float(max(err_w, err_b))}
