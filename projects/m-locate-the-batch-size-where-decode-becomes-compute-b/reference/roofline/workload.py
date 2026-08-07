from roofline.analysis import calculate_operational_intensity


def classify_workload_dominance(workload_profiles, model_config, hardware_specs):
    roofline_knee = hardware_specs["peak_flops"] / hardware_specs["peak_bandwidth"]
    classifications = []

    for profile in workload_profiles:
        prefill_batch = profile["prefill_batch"]
        prefill_len = profile["prefill_len"]
        decode_batch = profile["decode_batch"]
        decode_len = profile["decode_len"]

        intensity_prefill = calculate_operational_intensity(
            model_config, prefill_batch, prefill_len, phase="prefill"
        )
        intensity_decode = calculate_operational_intensity(
            model_config, decode_batch, decode_len, phase="decode"
        )

        if intensity_prefill >= roofline_knee and intensity_decode < roofline_knee:
            classifications.append("prefill_compute_decode_memory")
        elif intensity_prefill >= roofline_knee and intensity_decode >= roofline_knee:
            classifications.append("compute_heavy")
        elif intensity_prefill < roofline_knee and intensity_decode < roofline_knee:
            classifications.append("memory_heavy")
        else:
            classifications.append("prefill_memory_decode_compute")

    return classifications
