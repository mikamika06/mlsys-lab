import random

RANDOM_SEED = 42

def generate_fixtures():
    rng = random.Random(RANDOM_SEED)
    logs = []
    expected_parsed = []
    total_time = 0.0
    for i in range(10):
        prim = rng.choice(["convolution", "inner_product", "eltwise", "pooling"])
        impl = rng.choice(["jit:avx512", "jit:fma", "jit:ref", "ref:any"])
        t = round(rng.uniform(0.1, 5.5), 2)
        total_time += t
        line = f"onednn_verbose,info,exec,cpu,{prim},{impl},g0,mb1,1.23ms,{t}ms"
        logs.append(line)
        expected_parsed.append({
            "status": "info",
            "category": "exec",
            "engine": "cpu",
            "primitive": prim,
            "impl": impl,
            "time_ms": t
        })
    wall_time = total_time * 1.05
    return logs, expected_parsed, wall_time

SAMPLE_LOGS, EXPECTED_PARSED, SAMPLE_WALL_TIME = generate_fixtures()

def parse_row(line):
    parts = line.strip().split(",")
    if len(parts) < 10 or parts[0] != "onednn_verbose":
        return None
    return {
        "status": parts[1],
        "category": parts[2],
        "engine": parts[3],
        "primitive": parts[4],
        "impl": parts[5],
        "time_ms": float(parts[9].replace("ms", ""))
    }

def classify_and_reconcile(rows, wall_time):
    parsed = [parse_row(r) for r in rows if parse_row(r) is not None]
    classes = {}
    total_kernel_time = sum(p["time_ms"] for p in parsed)
    for p in parsed:
        impl = p["impl"]
        base = impl.split(":")[0]
        classes.setdefault(base, 0.0)
        classes[base] += p["time_ms"]
    ratio = total_kernel_time / wall_time if wall_time > 0 else 0.0
    return {
        "total_kernel_time_ms": round(total_kernel_time, 2),
        "wall_time_ms": round(wall_time, 2),
        "ratio": round(ratio, 4),
        "classes": {k: round(v, 2) for k, v in classes.items()}
    }
