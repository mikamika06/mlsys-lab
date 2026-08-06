import numpy as np


def compute_quant_params(float_min, float_max, qmin=0, qmax=255):
    scale = (float_max - float_min) / (qmax - qmin)
    if scale == 0:
        scale = 1.0
    zero_point = int(round(qmin - float_min / scale))
    zero_point = max(qmin, min(qmax, zero_point))
    return float(scale), int(zero_point)


def quantize_float_to_int8(x, scale, zero_point, qmin=0, qmax=255):
    q = np.round(x / scale) + zero_point
    return np.clip(q, qmin, qmax).astype(np.uint8)


def match_input_scale_zp(
    mean, std, f_min=0.0, f_max=1.0, qmin=0, qmax=255
):
    avg_mean = float(np.mean(mean))
    avg_std = float(np.mean(std))
    norm_min = (f_min - avg_mean) / avg_std
    norm_max = (f_max - avg_mean) / avg_std
    return compute_quant_params(norm_min, norm_max, qmin, qmax)


def diagnose_and_fix_layout(img, src_format, dst_format, src_order, dst_order):
    out = img.copy()
    if src_order != dst_order:
        if src_order == "BGR" and dst_order == "RGB":
            if src_format == "NHWC":
                out = out[..., ::-1]
            elif src_format == "NCHW":
                out = out[:, ::-1, :, :]
        elif src_order == "RGB" and dst_order == "BGR":
            if src_format == "NHWC":
                out = out[..., ::-1]
            elif src_format == "NCHW":
                out = out[:, ::-1, :, :]

    if src_format != dst_format:
        if src_format == "NHWC" and dst_format == "NCHW":
            out = np.transpose(out, (0, 3, 1, 2))
        elif src_format == "NCHW" and dst_format == "NHWC":
            out = np.transpose(out, (0, 2, 3, 1))

    return np.ascontiguousarray(out)


def fold_normalize_into_graph(weights, bias, mean, std):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    std_inv = 1.0 / std
    if weights.ndim == 4:
        w_folded = weights * std_inv[None, :, None, None]
    else:
        w_folded = weights * std_inv[None, :]

    b_folded = bias - np.sum(
        weights * (mean / std)[None, :, None, None], axis=(1, 2, 3)
    )

    return w_folded.astype(np.float32), b_folded.astype(np.float32)


def process_pipeline(
    raw_img,
    src_format,
    dst_format,
    src_order,
    dst_order,
    mean,
    std,
    weights,
    bias,
):
    fixed_img = diagnose_and_fix_layout(
        raw_img, src_format, dst_format, src_order, dst_order
    )
    folded_w, folded_b = fold_normalize_into_graph(weights, bias, mean, std)
    return fixed_img, (folded_w, folded_b)


TEST_CASES_QUANT = [
    {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "f_min": 0.0,
        "f_max": 1.0,
    },
    {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "f_min": 0.0,
        "f_max": 1.0,
    },
    {
        "mean": [123.68, 116.78, 103.94],
        "std": [58.393, 57.12, 57.375],
        "f_min": 0.0,
        "f_max": 255.0,
    },
]
