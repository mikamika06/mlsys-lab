def find_overfitting_onset_step(training_logs):
    if not training_logs:
        return None

    min_eval_loss = float("inf")
    onset_step = None

    for entry in training_logs:
        step = entry.get("step")
        eval_loss = entry.get("eval_loss")
        if eval_loss is None or step is None:
            continue
        if eval_loss < min_eval_loss:
            min_eval_loss = eval_loss
            onset_step = step

    return onset_step
