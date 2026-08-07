import numpy as np


def predict_amx(m, n, k, dtype):
    scale = 2.0 if dtype == "int8" else 1.0
    pad_m = (16 - (m % 16)) % 16
    pad_n = (16 - (n % 16)) % 16
    pad_k = (32 - (k % 32)) % 32 if dtype == "int8" else (16 - (k % 16)) % 16
    eff = (m * n * k) / ((m + pad_m) * (n + pad_n) * (k + pad_k) + 1e-5)
    return float(1000.0 * scale * eff)


def predict_avx512(m, n, k, dtype):
    scale = 1.5 if dtype == "int8" else 1.0
    util = 1.0 - 0.05 * ((n % 16) != 0)
    return float(400.0 * scale * util * (m * n * k) ** 0.02)


def analyze_shape(m, n, k, dtype):
    return {
        "amx": predict_amx(m, n, k, dtype),
        "avx512": predict_avx512(m, n, k, dtype)
    }


def compare_with_measurement(m, n, k, dtype, measured):
    pred = predict_amx(m, n, k, dtype)
    err = abs(pred - measured) / (measured + 1e-5)
    return err <= 0.15


def find_crossover(shapes, dtype):
    best = shapes[0]
    min_diff = float("inf")
    for s in shapes:
        mn, nn, kn = s
        diff = abs(predict_amx(mn, nn, kn, dtype) - predict_avx512(mn, nn, kn, dtype))
        if diff < min_diff:
            min_diff = diff
            best = s
    return best


def select_best_isa(m, n, k, dtype):
    amx = predict_amx(m, n, k, dtype)
    avx = predict_avx512(m, n, k, dtype)
    return "amx" if amx >= avx else "avx512"


def get_test_shapes():
    return [
        (16, 16, 64, "int8"),
        (32, 32, 64, "int8"),
        (64, 64, 128, "int8"),
        (128, 128, 256, "int8"),
        (256, 256, 256, "int8"),
        (16, 16, 16, "bf16"),
        (32, 32, 32, "bf16"),
        (64, 64, 64, "bf16"),
        (128, 128, 128, "bf16"),
        (256, 256, 256, "bf16")
    ]
