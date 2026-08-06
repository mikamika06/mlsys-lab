import math

WORKLOAD_NAMES = [
    "attn_b1_s128",
    "attn_b4_s512",
    "mlp_b1_d4096",
    "conv_b2_c64",
    "norm_b8_d1024",
    "rope_b1_s256",
]

EXPECTED_CALLS = {
    "attn_b1_s128": 150,
    "attn_b4_s512": 30,
    "mlp_b1_d4096": 500,
    "conv_b2_c64": 10,
    "norm_b8_d1024": 200,
    "rope_b1_s256": 1000,
}


def generate_records():
    records = []
    base_data = [
        ("attn_b1_s128", 150.0, [12.0, 2.0, 2.0, 2.0], 5.0, [3.0, 3.0, 3.0]),
        ("attn_b4_s512", 400.0, [50.0, 10.0, 10.0], 10.0, [15.0, 15.0, 15.0]),
        ("mlp_b1_d4096", 80.0, [8.0, 1.5, 1.5], 2.0, [1.8, 1.8, 1.8]),
        ("conv_b2_c64", 200.0, [20.0, 4.0, 4.0], 15.0, [3.5, 3.5, 3.5]),
        ("norm_b8_d1024", 50.0, [5.0, 0.5, 0.5], 1.0, [0.6, 0.6, 0.6]),
        ("rope_b1_s256", 30.0, [3.0, 0.2, 0.2], 0.5, [0.25, 0.25, 0.25]),
    ]
    for w, j_comp, j_execs, a_load, a_execs in base_data:
        records.append({
            "workload": w,
            "mode": "jit",
            "compile_ms": j_comp,
            "exec_ms": j_execs,
        })
        records.append({
            "workload": w,
            "mode": "aot",
            "load_ms": a_load,
            "exec_ms": a_execs,
        })
    return records


def parse_overhead_records(records):
    grouped = {}
    for r in records:
        w = r["workload"]
        if w not in grouped:
            grouped[w] = {
                "jit_compile_ms": 0.0,
                "jit_execs": [],
                "aot_load_ms": 0.0,
                "aot_execs": [],
            }
        mode = r.get("mode", "jit")
        if mode == "jit":
            grouped[w]["jit_compile_ms"] += float(r.get("compile_ms", 0.0))
            execs = r.get("exec_ms", [])
            if isinstance(execs, (int, float)):
                execs = [float(execs)]
            if len(execs) > 1:
                grouped[w]["jit_execs"].extend([float(x) for x in execs[1:]])
            elif len(execs) == 1:
                grouped[w]["jit_execs"].extend([float(x) for x in execs])
        elif mode == "aot":
            grouped[w]["aot_load_ms"] += float(r.get("load_ms", 0.0))
            execs = r.get("exec_ms", [])
            if isinstance(execs, (int, float)):
                execs = [float(execs)]
            if len(execs) > 1:
                grouped[w]["aot_execs"].extend([float(x) for x in execs[1:]])
            elif len(execs) == 1:
                grouped[w]["aot_execs"].extend([float(x) for x in execs])

    out = {}
    for w, data in grouped.items():
        j_exec = sum(data["jit_execs"]) / len(data["jit_execs"]) if data["jit_execs"] else 0.0
        a_exec = sum(data["aot_execs"]) / len(data["aot_execs"]) if data["aot_execs"] else 0.0
        out[w] = {
            "jit_compile_ms": float(data["jit_compile_ms"]),
            "jit_exec_ms": float(j_exec),
            "aot_load_ms": float(data["aot_load_ms"]),
            "aot_exec_ms": float(a_exec),
        }
    return out


def compute_breakeven(profile):
    j_comp = profile["jit_compile_ms"]
    j_exec = profile["jit_exec_ms"]
    a_load = profile["aot_load_ms"]
    a_exec = profile["aot_exec_ms"]

    if j_exec < a_exec:
        preferred = "jit"
        setup_diff = j_comp - a_load
        exec_diff = a_exec - j_exec
    elif a_exec < j_exec:
        preferred = "aot"
        setup_diff = a_load - j_comp
        exec_diff = j_exec - a_exec
    else:
        preferred = "jit" if j_comp <= a_load else "aot"
        setup_diff = 0.0
        exec_diff = 0.0

    if exec_diff <= 0.0 or setup_diff <= 0.0:
        n_break = 1.0
    else:
        n_break = float(max(1, math.ceil(setup_diff / exec_diff)))

    if preferred == "jit":
        latency = j_comp + n_break * j_exec
    else:
        latency = a_load + n_break * a_exec

    return {
        "preferred_mode": preferred,
        "break_even_calls": float(n_break),
        "crossover_latency_ms": float(latency),
        "overhead_delta_ms": float(abs(j_comp - a_load)),
    }


def select_strategy(profiles, expected_calls):
    out = {}
    for w, p in profiles.items():
        n = expected_calls.get(w, 1)
        jit_tot = p["jit_compile_ms"] + n * p["jit_exec_ms"]
        aot_tot = p["aot_load_ms"] + n * p["aot_exec_ms"]
        if jit_tot <= aot_tot:
            mode = "jit"
            est = jit_tot
        else:
            mode = "aot"
            est = aot_tot
        out[w] = {
            "selected_mode": mode,
            "estimated_latency_ms": float(est),
            "savings_ms": float(abs(jit_tot - aot_tot)),
        }
    return out
