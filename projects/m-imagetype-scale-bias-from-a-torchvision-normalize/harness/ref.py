import numpy as np

TEST_CONFIGS = [
    {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    },
    {
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
    },
]


def oracle_scale_bias(mean, std):
    m = np.array(mean, dtype=np.float64)
    s = np.array(std, dtype=np.float64)
    scale = (1.0 / (255.0 * s)).tolist()
    bias = (-m / s).tolist()
    return scale, bias


def generate_cnn_outputs():
    np.random.seed(42)
    torch_out = np.random.randn(2, 64, 14, 14).astype(np.float32)
    noise = np.random.randn(2, 64, 14, 14).astype(np.float32) * 1e-6
    coreml_out = torch_out + noise
    return torch_out, coreml_out


def generate_package_specs():
    fp32_spec = {
        "metadata_bytes": 4096,
        "weights": {
            "conv1.weight": 1024 * 1024 * 4,
            "conv2.weight": 2048 * 2048 * 4,
            "fc.weight": 512 * 1000 * 4,
        },
    }
    fp16_spec = {
        "metadata_bytes": 4096,
        "weights": {
            "conv1.weight": 1024 * 1024 * 2,
            "conv2.weight": 2048 * 2048 * 2,
            "fc.weight": 512 * 1000 * 2,
        },
    }
    return fp32_spec, fp16_spec
