def quality_penalty(drop_rate: float) -> float:
    return 1.0 - (1.0 - drop_rate)**3
