def attribute_regression(baseline_profile, new_profile):
    attributions = {}
    for k in new_profile:
        diff = new_profile[k] - baseline_profile.get(k, 0)
        attributions[k] = {"delta": diff, "ratio": float(diff) / float(baseline_profile[k]) if baseline_profile.get(k, 0) > 0 else 0.0}
    return attributions
