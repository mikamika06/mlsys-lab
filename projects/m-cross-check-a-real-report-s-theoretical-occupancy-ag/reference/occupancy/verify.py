from occupancy.compute import compute_theoretical_occupancy
from occupancy.sections import map_metric_to_section


def cross_check_occupancy(report_data: dict, device_props: dict) -> dict:
    regs = report_data.get("regs_per_thread", 32)
    smem = report_data.get("smem_per_block", 1024)
    block_size = report_data.get("block_size", 256)
    reported_occ = report_data.get("reported_occupancy", 0.5)

    computed_occ = compute_theoretical_occupancy(regs, smem, block_size, device_props)

    rel_err = abs(reported_occ - computed_occ) / (computed_occ if computed_occ > 0 else 1.0)

    return {
        "computed_occupancy": computed_occ,
        "reported_occupancy": reported_occ,
        "relative_error": rel_err,
        "match": rel_err < 0.05
    }
