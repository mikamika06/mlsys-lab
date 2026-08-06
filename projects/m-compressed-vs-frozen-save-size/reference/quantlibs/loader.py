import time

def compare_load_times(formats, file_paths):
    results = {}
    for fmt, path in zip(formats, file_paths):
        t0 = time.perf_counter()
        dummy_sum = 0
        with open(path, "rb") as f:
            chunk = f.read(1024)
            while chunk:
                dummy_sum += sum(chunk)
                chunk = f.read(1024)
        t1 = time.perf_counter()
        results[fmt] = max(0.000001, t1 - t0)
    sorted_formats = sorted(results.keys(), key=lambda x: results[x])
    return {"times": results, "fastest": sorted_formats[0]}
