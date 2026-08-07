import ref


def check(workdir):
    from quantizer.oneshot import run_oneshot
    from quantizer.onloading import evaluate_onloading_impact

    model = ref.get_tiny_model()
    quantized = run_oneshot(model, sequential_onloading=True)
    ratio = ref.compute_size_ratio(model, quantized)

    impact = evaluate_onloading_impact(model)

    size_ok = 1.0 if ratio <= 0.6 else 0.0
    onloading_ok = 1.0 if isinstance(impact, dict) and "peak_memory_off" in impact and "peak_memory_on" in impact else 0.0

    return {
        "size_ratio": float(ratio),
        "onloading_correct": float(onloading_ok)
    }
