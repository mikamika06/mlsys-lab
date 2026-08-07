import numpy as np


def build_isolation_command(onnx_path, trt_path, layer_names, output_json):
    cmd = [
        "polygraphy", "run", onnx_path,
        "--onnxrt",
        "--trt", trt_path,
        "--onnx-outputs"
    ]
    cmd.extend(layer_names)
    cmd.extend([
        "--trt-outputs"
    ])
    cmd.extend(layer_names)
    cmd.extend([
        "--save-inspect-iter-info", output_json
    ])
    return cmd


def build_mark_all_command(model_path, output_model_path):
    return [
        "polygraphy", "surgeon", "sanitize", model_path,
        "--override-outputs", "mark", "all",
        "-o", output_model_path
    ]


def compute_mae(a, b):
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a_arr - b_arr)))


def compute_max_abs_diff(a, b):
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a_arr - b_arr)))


def compute_rel_error(a, b, eps=1e-7):
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a_arr - b_arr) / (np.abs(b_arr) + eps)))


def compute_snr(a, b, eps=1e-10):
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    signal_pow = np.sum(b_arr ** 2)
    noise_pow = np.sum((a_arr - b_arr) ** 2)
    if noise_pow < eps:
        return float("inf")
    return float(10.0 * np.log10(signal_pow / (noise_pow + eps)))


def compute_polygraphy_stats(a, b):
    return {
        "mae": compute_mae(a, b),
        "max_abs_diff": compute_max_abs_diff(a, b),
        "rel_error": compute_rel_error(a, b),
        "snr_db": compute_snr(a, b)
    }


def find_first_divergent_layer(layer_outputs, rtol=1e-3, atol=1e-5):
    for layer_name, (out_ref, out_candidate) in layer_outputs.items():
        ref_arr = np.asarray(out_ref, dtype=np.float64)
        cand_arr = np.asarray(out_candidate, dtype=np.float64)
        if not np.allclose(ref_arr, cand_arr, rtol=rtol, atol=atol):
            return layer_name
    return None


TEST_CASES_COMMANDS = [
    {
        "onnx": "models/resnet50.onnx",
        "trt": "models/resnet50.engine",
        "layers": ["conv1", "relu1", "maxpool1"],
        "out_json": "results/resnet_iso.json"
    },
    {
        "onnx": "models/bert.onnx",
        "trt": "models/bert.engine",
        "layers": ["embeddings", "encoder_layer_0", "encoder_layer_1"],
        "out_json": "results/bert_iso.json"
    }
]

TEST_CASES_DIVERGENCE = [
    {
        "layers": {
            "node_0_conv": (np.array([1.0, 2.0, 3.0]), np.array([1.0001, 2.0001, 3.0001])),
            "node_1_relu": (np.array([0.0, 1.0, 2.0]), np.array([0.0000, 1.0000, 2.0000])),
            "node_2_matmul": (np.array([10.0, 20.0]), np.array([10.5, 20.8])),
            "node_3_add": (np.array([5.0, 5.0]), np.array([10.0, 10.0]))
        },
        "rtol": 1e-3,
        "atol": 1e-4
    },
    {
        "layers": {
            "layer_a": (np.array([0.1, 0.2]), np.array([0.1, 0.2])),
            "layer_b": (np.array([0.3, 0.4]), np.array([0.3, 0.4]))
        },
        "rtol": 1e-2,
        "atol": 1e-2
    }
]

TEST_CASES_STATS = [
    {
        "a": np.array([1.0, 2.0, 3.0, 4.0]),
        "b": np.array([1.1, 1.9, 3.2, 3.8])
    },
    {
        "a": np.array([0.0, 0.5, 1.0]),
        "b": np.array([0.01, 0.49, 1.02])
    }
]
