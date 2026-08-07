def track_activation_memory(schedule_events: list, num_stages: int) -> list:
    raise NotImplementedError

def is_within_memory_budget(schedule_events: list, num_stages: int, max_units: int) -> bool:
    raise NotImplementedError
