from pipelib.schedule import Schedule1F1B, ScheduleGPipe


def measure_peak_inflight_microbatches(p: int, m: int, v: int = 1, schedule_type: str = "1f1b") -> list[int]:
    """Measure peak in-flight microbatches per physical rank."""
    if schedule_type == "gpipe":
        sim = ScheduleGPipe(p, m)
        res = sim.run()
        peaks = [0] * p
        for stage in range(p):
            events = res["events"][stage]
            current = 0
            max_c = 0
            for ev in sorted(events, key=lambda x: x["start"]):
                if ev["type"] == "forward":
                    current += 1
                elif ev["type"] == "backward":
                    current -= 1
                if current > max_c:
                    max_c = current
            peaks[stage] = max_c
        return peaks

    if schedule_type == "1f1b":
        sim = Schedule1F1B(p, m)
        res = sim.run()
        peaks = [0] * p
        for stage in range(p):
            events = res["events"][stage]
            current = 0
            max_c = 0
            for ev in sorted(events, key=lambda x: x["start"]):
                if ev["type"] == "forward":
                    current += 1
                elif ev["type"] == "backward":
                    current -= 1
                if current > max_c:
                    max_c = current
            peaks[stage] = max_c
        return peaks

    if schedule_type == "interleaved":
        num_virtual = p * v
        warmup_limit = p
        peaks = [0] * p
        for stage in range(p):
            val = min(m, warmup_limit + (p - 1 - stage))
            peaks[stage] = val
        return peaks

    raise ValueError(f"Unknown schedule_type: {schedule_type}")


def estimate_activation_memory_mb(peak_inflight: list[int], bytes_per_microbatch: float) -> list[float]:
    """Calculate activation memory per rank based on peak in-flight microbatches."""
    mb_factor = 1024.0 * 1024.0
    return [(c * bytes_per_microbatch) / mb_factor for c in peak_inflight]
