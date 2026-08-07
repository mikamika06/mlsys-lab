def compute_accuracy_curve(ratios: list[float], eval_fn) -> list[tuple[float, float]]:
    return [(float(r), float(eval_fn(r))) for r in ratios]
