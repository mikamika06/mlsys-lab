from delegation.fraction import measure_delegated_fraction

def compare_partitioners(model):
    xnn_frac = measure_delegated_fraction(model, "XNNPACK")
    core_frac = measure_delegated_fraction(model, "CoreML")
    return {
        "xnnpack_fraction": xnn_frac,
        "coreml_fraction": core_frac,
        "preferred": "XNNPACK" if xnn_frac >= core_frac else "CoreML"
    }
