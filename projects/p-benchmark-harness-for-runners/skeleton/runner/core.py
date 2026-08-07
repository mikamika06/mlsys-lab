def measure_latency(raw_runs, warmup_count=1):
    raise NotImplementedError


def split_phases(prefill_times, decode_times):
    raise NotImplementedError


def required_samples(std_dev, target_width, confidence_z=1.96):
    raise NotImplementedError


def correct_thermal(raw_speeds, degradation_rate=0.01):
    raise NotImplementedError


def check_consistency(run_intervals):
    raise NotImplementedError


def generate_ci_report(metrics_dict):
    raise NotImplementedError
