import numpy as np


def select_opt_shape(batch_samples, strategy="p50"):
    samples = np.asarray(batch_samples, dtype=np.int64)
    if strategy == "p50":
        return int(np.percentile(samples, 50))
    elif strategy == "p90":
        return int(np.percentile(samples, 90))
    elif strategy == "mode":
        vals, counts = np.unique(samples, return_counts=True)
        return int(vals[np.argmax(counts)])
    elif strategy == "mean":
        return int(np.round(np.mean(samples)))
    raise ValueError(f"Unknown strategy {strategy}")


def calculate_profile_bounds(batch_samples, strategy="p50", padding_ratio=0.1):
    samples = np.asarray(batch_samples, dtype=np.int64)
    opt_val = select_opt_shape(samples, strategy=strategy)
    min_val = int(np.min(samples))
    max_val = int(np.max(samples))
    if padding_ratio > 0:
        pad = int(np.ceil((max_val - min_val) * padding_ratio))
        min_val = max(1, min_val - pad)
        max_val = max_val + pad
    min_val = min(min_val, opt_val)
    max_val = max(max_val, opt_val)
    return (min_val, opt_val, max_val)


def build_profile_plan(workload_spec):
    profiles = []
    for prof_spec in workload_spec.get("profiles", []):
        prof_dict = {}
        strategy = prof_spec.get("strategy", "p50")
        padding = prof_spec.get("padding_ratio", 0.1)
        for tensor_name, data in prof_spec.get("tensors", {}).items():
            fixed_dims = data.get("fixed_dims", [])
            dynamic_samples = data.get("samples", [])
            min_d, opt_d, max_d = calculate_profile_bounds(dynamic_samples, strategy=strategy, padding_ratio=padding)
            prof_dict[tensor_name] = {
                "min": tuple(fixed_dims + [min_d]),
                "opt": tuple(fixed_dims + [opt_d]),
                "max": tuple(fixed_dims + [max_d]),
            }
        profiles.append(prof_dict)
    return profiles


def make_workload_fixtures():
    np.random.seed(42)
    fixtures = []
    strategies = ["p50", "p90", "mode", "mean"]
    for i in range(4):
        samples_a = np.random.randint(1, 128, size=100).tolist()
        samples_b = np.random.randint(16, 256, size=100).tolist()
        spec = {
            "profiles": [
                {
                    "strategy": strategies[i],
                    "padding_ratio": 0.1,
                    "tensors": {
                        "input_ids": {"fixed_dims": [1], "samples": samples_a},
                        "attention_mask": {"fixed_dims": [1], "samples": samples_b}
                    }
                }
            ]
        }
        fixtures.append(spec)
    return fixtures
