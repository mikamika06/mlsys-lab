from quant.calib import compute_scales
from quant.eval import Evaluator
from quant.recipes import quantize_int4, quantize_int8
from quant.sensitivity import (
    compute_layer_sensitivity,
    select_mixed_precision_config,
)


def run_quantization_pipeline(model, dataset, calibration_data):
    """Runs complete quantized model generation pipeline."""
    evaluator = Evaluator(dataset)
    base_acc = evaluator.evaluate(model)
    sensitivities = compute_layer_sensitivity(model, evaluator, calibration_data)
    config = select_mixed_precision_config(sensitivities)

    quantized_weights = {}
    total_orig_bits = 0
    total_quant_bits = 0

    for name, w in model.layers.items():
        bits = config.get(name, 4)
        total_orig_bits += w.size * 32
        total_quant_bits += w.size * bits
        if bits == 8:
            _, scale, deq = quantize_int8(w)
        else:
            scale = compute_scales(w, len(calibration_data), calibration_data)
            _, scale, deq = quantize_int4(w, scales=scale)
        model.layers[name] = deq
        quantized_weights[name] = deq

    new_acc = evaluator.evaluate(model)
    compression_ratio = float(total_orig_bits / total_quant_bits)

    return {
        "baseline_accuracy": float(base_acc),
        "quantized_accuracy": float(new_acc),
        "accuracy_drop": float(base_acc - new_acc),
        "compression_ratio": float(compression_ratio),
        "config": config,
    }
