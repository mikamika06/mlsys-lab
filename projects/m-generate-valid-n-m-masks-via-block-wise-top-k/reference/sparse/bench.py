import re


def parse_a100_gemm_log(log_content):
    results = []
    for line in log_content.strip().split("\n"):
        if not line.strip():
            continue
        data = {}
        parts = line.split("|")
        for part in parts:
            if ":" in part:
                k, v = part.split(":", 1)
                data[k.strip().lower()] = v.strip()
        if not data:
            match_tflops = re.findall(r"([\d\.]+)\s*TFLOPS", line)
            match_speedup = re.search(r"([\d\.]+)x", line)
            results.append({
                "raw": line.strip(),
                "tflops": [float(x) for x in match_tflops],
                "speedup": float(match_speedup.group(1)) if match_speedup else 0.0
            })
        else:
            results.append(data)
    return results
