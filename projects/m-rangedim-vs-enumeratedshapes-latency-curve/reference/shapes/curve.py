import numpy as np


def latency_ratio_curve(shape_ranges, profile_data):
    ratios = []
    for length in shape_ranges:
        range_lat = profile_data.get(("rangedim", length), 1.0)
        enum_lat = profile_data.get(("enumerated", length), 1.0)
        ratio = range_lat / (enum_lat + 1e-6)
        ratios.append((length, float(ratio)))
    return ratios
