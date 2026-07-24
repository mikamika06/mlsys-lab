def compute_comm_volume(phi: float, stage: int) -> float:
    if stage not in (2, 3):
        raise ValueError("unsupported stage")

    volume = phi + phi
    if stage == 3:
        volume += phi
    return float(volume)
