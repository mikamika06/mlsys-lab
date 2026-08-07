def cross_check_report(report, computed_occ):
    return abs(report["theoretical_occupancy"] - computed_occ) < 0.05
