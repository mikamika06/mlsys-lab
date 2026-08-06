def detect_contaminated_baseline(
    teacher_data: dict,
    historical_ppl: dict,
    max_perplexity_drop_ratio: float = 0.4,
    min_entropy_threshold: float = 1e-3,
) -> dict:
    raise NotImplementedError
