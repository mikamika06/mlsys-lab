import re


def parse_kernel_config(kernel_code: str) -> dict:
    """Parse tuning parameters from a TorchInductor Triton kernel string."""
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
