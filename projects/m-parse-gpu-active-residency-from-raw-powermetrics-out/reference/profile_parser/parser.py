import re


def parse_gpu_active_residency(powermetrics_text: str) -> list[float]:
    """Parse active residency percentage for each sample interval in powermetrics text."""
    samples = powermetrics_text.split("*** Sampled system activity")
    results = []

    for sample in samples:
        if not sample.strip():
            continue
        gpu_match = re.search(r"GPU\s+HW\s+active\s+residency:\s*([\d\.]+)%", sample, re.IGNORECASE)
        if gpu_match:
            results.append(float(gpu_match.group(1)))
            continue

        idle_match = re.search(r"GPU\s+HW\s+idle\s+residency:\s*([\d\.]+)%", sample, re.IGNORECASE)
        if idle_match:
            idle_pct = float(idle_match.group(1))
            results.append(max(0.0, min(100.0, 100.0 - idle_pct)))
            continue

        active_freq_match = re.search(r"GPU\s+active\s+frequency:\s*[\d\.]+\s*MHz\s*\(([\d\.]+)%\)", sample, re.IGNORECASE)
        if active_freq_match:
            results.append(float(active_freq_match.group(1)))
            continue

        use_match = re.search(r"GPU\s+use:\s*([\d\.]+)%", sample, re.IGNORECASE)
        if use_match:
            results.append(float(use_match.group(1)))

    return results


def parse_ane_power_mw(powermetrics_text: str) -> list[float]:
    """Parse ANE power in milliwatts for each sample interval in powermetrics text."""
    samples = powermetrics_text.split("*** Sampled system activity")
    results = []

    for sample in samples:
        if not sample.strip():
            continue
        ane_match = re.search(r"ANE\s+Power:\s*([\d\.]+)\s*(mW|W)", sample, re.IGNORECASE)
        if ane_match:
            val = float(ane_match.group(1))
            unit = ane_match.group(2).lower()
            if unit == "w":
                val *= 1000.0
            results.append(val)

    return results
