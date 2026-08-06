import os


def measure_footprint(model_sizes, base_dir):
    results = {}
    for size in model_sizes:
        size_dir = os.path.join(base_dir, f"model_{size}")
        os.makedirs(size_dir, exist_ok=True)
        dummy_file = os.path.join(size_dir, "kernel.so")
        content = b"X" * (size * 1024)
        with open(dummy_file, "wb") as f:
            f.write(content)
        total_bytes = sum(
            os.path.getsize(os.path.join(root, file))
            for root, _, files in os.walk(size_dir)
            for file in files
        )
        results[size] = total_bytes
    return results
