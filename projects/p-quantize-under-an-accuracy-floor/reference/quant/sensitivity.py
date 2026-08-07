from quant.recipes import quantize_int4


def compute_layer_sensitivity(model, evaluator, calibration_data):
    """Calculates sensitivity per layer by measuring accuracy drop under quantization."""
    baseline_acc = evaluator.evaluate(model)
    sensitivities = {}
    for name, w in model.layers.items():
        orig_w = w.copy()
        _, _, deq = quantize_int4(w)
        model.layers[name] = deq
        acc = evaluator.evaluate(model)
        sensitivities[name] = float(baseline_acc - acc)
        model.layers[name] = orig_w
    return sensitivities


def select_mixed_precision_config(sensitivity_scores, target_bits=5.5):
    """Selects bit-width per layer according to sensitivity rankings."""
    sorted_layers = sorted(
        sensitivity_scores.items(), key=lambda item: item[1], reverse=True
    )
    num_high = max(1, int(len(sorted_layers) * 0.25))
    config = {}
    for i, (name, _) in enumerate(sorted_layers):
        if i < num_high:
            config[name] = 8
        else:
            config[name] = 4
    return config
