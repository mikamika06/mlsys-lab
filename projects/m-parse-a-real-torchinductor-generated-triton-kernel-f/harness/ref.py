import re

SAMPLE_KERNELS = [
    """
@triton_heuristics.pointwise(
    size_hints=[1024],
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'axises': {}},
    configs=[{'XBLOCK': 128}],
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    pass
""",
    """
@triton_heuristics.reduction(
    size_hints=[2048, 1024],
    filename=__file__,
    triton_meta={'num_warps': 8, 'num_stages': 3, 'constants': {}},
    configs=[{'XBLOCK': 32, 'RBLOCK': 256}],
)
@triton.jit
def triton_red_fused_1(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    pass
""",
    """
triton_meta = {'num_warps': 2, 'num_stages': 1}
@triton.jit
def triton_custom_2(in_ptr, out_ptr, XBLOCK=64, YBLOCK=16):
    pass
""",
    """
@triton_heuristics.grid(1024)
@triton.jit
def triton_grid_3(in_ptr, out_ptr, XBLOCK: tl.constexpr = 256):
    pass
""",
    """
@triton_heuristics.template(
    configs=[transform_config({'XBLOCK': 512, 'YBLOCK': 32, 'RBLOCK': 64}, num_warps=16, num_stages=5)]
)
@triton.jit
def triton_tmpl_4(in_ptr, out_ptr):
    pass
"""
]


def parse_kernel_config(kernel_code: str) -> dict:
    config = {}

    triton_meta_match = re.search(r"triton_meta\s*=\s*(\{.*?\})", kernel_code, re.DOTALL)
    if triton_meta_match:
        meta_str = triton_meta_match.group(1)
        for key in ["num_warps", "num_stages"]:
            m = re.search(rf"'{key}'\s*:\s*(\d+)", meta_str)
            if m:
                config[key] = int(m.group(1))

    triton_heuristics_match = re.search(r"@triton_heuristics\.\w+\(\s*configs=\s*\[(.*?)\]", kernel_code, re.DOTALL)
    if triton_heuristics_match:
        configs_str = triton_heuristics_match.group(1)
        for param in ["XBLOCK", "YBLOCK", "ZBLOCK", "RBLOCK"]:
            m = re.search(rf"'{param}'\s*:\s*(\d+)", configs_str)
            if m:
                config[param] = int(m.group(1))
            else:
                m_kw = re.search(rf"{param}\s*=\s*(\d+)", configs_str)
                if m_kw:
                    config[param] = int(m_kw.group(1))

    for param in ["XBLOCK", "YBLOCK", "ZBLOCK", "RBLOCK"]:
        if param not in config:
            m = re.search(rf"\b{param}\s*:\_?\s*tl\.constexpr\s*=\s*(\d+)", kernel_code)
            if not m:
                m = re.search(rf"\b{param}\s*=\s*(\d+)", kernel_code)
            if m:
                config[param] = int(m.group(1))

    if "num_warps" not in config:
        m = re.search(r"num_warps\s*=\s*(\d+)", kernel_code)
        if m:
            config["num_warps"] = int(m.group(1))

    if "num_stages" not in config:
        m = re.search(r"num_stages\s*=\s*(\d+)", kernel_code)
        if m:
            config["num_stages"] = int(m.group(1))

    defaults = {"num_warps": 4, "num_stages": 2}
    for k, v in defaults.items():
        if k not in config:
            config[k] = v

    return config


def diff_configs(default_config: dict, autotune_config: dict) -> dict:
    all_keys = set(default_config.keys()) | set(autotune_config.keys())
    changed = {}
    same = {}

    for k in sorted(all_keys):
        def_val = default_config.get(k)
        auto_val = autotune_config.get(k)
        if def_val != auto_val:
            changed[k] = {"default": def_val, "autotune": auto_val}
        else:
            same[k] = def_val

    return {"changed": changed, "same": same}


def find_argmin_config(candidate_logs: list) -> dict:
    if not candidate_logs:
        return {}

    valid_candidates = [c for c in candidate_logs if c.get("status") == "OK" and "time_ms" in c]
    if not valid_candidates:
        return {}

    best = min(valid_candidates, key=lambda x: x["time_ms"])
    return {
        "config": best["config"],
        "time_ms": best["time_ms"],
        "num_candidates_evaluated": len(candidate_logs),
    }


CANDIDATE_LOGS = [
    {"config": {"XBLOCK": 32, "num_warps": 4}, "time_ms": 1.25, "status": "OK"},
    {"config": {"XBLOCK": 64, "num_warps": 4}, "time_ms": 0.82, "status": "OK"},
    {"config": {"XBLOCK": 128, "num_warps": 8}, "time_ms": 0.45, "status": "OK"},
    {"config": {"XBLOCK": 256, "num_warps": 8}, "time_ms": 0.61, "status": "OK"},
    {"config": {"XBLOCK": 512, "num_warps": 16}, "time_ms": 0.0, "status": "FAILED"},
]
