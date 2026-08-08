def select_best_calibration(datasets, eval_fn):
    best_name = None
    best_scale = 0.02
    min_err = float("inf")

    scales = [0.01, 0.02, 0.05, 0.10]
    for name, ds in datasets.items():
        for scale in scales:
            err = eval_fn(ds, scale)
            if err < min_err:
                min_err = err
                best_name = name
                best_scale = scale

    return best_name, best_scale, min_err
