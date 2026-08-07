def gpipe_bubble_fraction(num_stages: int, num_microbatches: int) -> float:
    raise NotImplementedError

def gpipe_peak_activation_units(num_stages: int, num_microbatches: int, stage_idx: int) -> int:
    raise NotImplementedError

def calculate_schedule_metrics(schedule_events: list, num_stages: int, num_microbatches: int) -> dict:
    raise NotImplementedError
