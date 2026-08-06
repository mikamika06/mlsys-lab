import random
import numpy as np


def generate_cases():
    rng = random.Random(42)
    cases = []
    for _ in range(5):
        reserved = rng.randint(20000000, 50000000)
        allocated = rng.randint(10000000, reserved - 5000000)
        free_blocks = [rng.randint(100000, 2000000) for _ in range(3)]
        max_free = max(free_blocks)
        total_free = sum(free_blocks)
        frag = 1.0 - (max_free / total_free) if total_free > 0 else 0.0
        req = rng.randint(max_free + 100000, total_free + 5000000)
        dump = f"total_reserved: {reserved}\ntotal_allocated: {allocated}\n"
        for fb in free_blocks:
            dump += f"free_block: {fb}\n"
        dump += f"requested_allocation: {req}\n"
        cases.append({
            "dump": dump,
            "expected_frag": float(frag),
            "expected_req": int(req)
        })
    return cases


def generate_history_cases():
    rng = random.Random(42)
    cases = []
    sites = ["attn_fwd", "attn_bwd", "mlp_fwd", "mlp_bwd", "optimizer_step"]
    for _ in range(5):
        records = []
        max_bytes = -1
        best_site = sites[0]
        for s in sites:
            b = rng.randint(1000000, 10000000)
            if b > max_bytes:
                max_bytes = b
                best_site = s
            records.append({"site": s, "bytes": b})
        cases.append({
            "snapshot": records,
            "expected_site": best_site
        })
    return cases


def generate_trend_cases():
    rng = random.Random(42)
    cases = []
    for _ in range(5):
        capacity = 100000000
        start_mem = 50000000
        slope = rng.randint(200000, 800000)
        steps = list(range(1, 30))
        mems = [start_mem + s * slope + rng.randint(-10000, 10000) for s in steps]
        x = np.array(steps, dtype=float)
        y = np.array(mems, dtype=float)
        poly = np.polyfit(x, y, 1)
        expected_step = int(np.round((capacity - poly[1]) / poly[0]))
        cases.append({
            "steps": steps,
            "mems": mems,
            "capacity": capacity,
            "expected_step": expected_step
        })
    return cases


REF_CASES_M1 = generate_cases()
REF_CASES_M2 = generate_history_cases()
REF_CASES_M3 = generate_trend_cases()
