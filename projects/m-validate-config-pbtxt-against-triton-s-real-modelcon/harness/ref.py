CONFIGS = [
    {
        "text": 'name: "model_a"\nplatform: "tensorflow_graphdef"\nmax_batch_size: 4',
        "expected_valid": True,
        "expected_errors": []
    },
    {
        "text": 'platform: "tensorflow_graphdef"\nmax_batch_size: 4',
        "expected_valid": False,
        "expected_errors": ["missing required field: name"]
    },
    {
        "text": 'name: "model_b"\nmax_batch_size: 8',
        "expected_valid": False,
        "expected_errors": ["missing required field: platform or backend"]
    },
    {
        "text": 'name: "model_c"\nbackend: "onnxruntime"\nmax_batch_size: -1',
        "expected_valid": False,
        "expected_errors": ["max_batch_size cannot be negative"]
    }
]

LAYOUT_CASES = [
    ("missing_config", "missing_config_pbtxt"),
    ("missing_versions", "missing_version_directories"),
    ("malformed_version", "malformed_version_directory_name"),
    ("valid_layout", "valid")
]

VERSION_CASES = [
    ([1, 2, 3, 4], {"latest": {"count": 2}}, [3, 4]),
    ([10, 20, 30], {"all": {}}, [10, 20, 30]),
    ([5, 12, 8], {"specific": {"versions": [5, 12, 99]}}, [5, 12])
]
