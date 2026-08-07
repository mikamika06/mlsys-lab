def compute_unrecovered_curve(ratios):
    return [max(0.0, 1.0 - 0.5 * (r ** 1.5)) for r in ratios]
