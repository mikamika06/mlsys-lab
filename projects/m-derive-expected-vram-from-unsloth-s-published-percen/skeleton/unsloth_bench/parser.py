"""Console log parser and metrics computation."""


def parse_unsloth_log(log_text: str) -> dict:
    """Parse Unsloth console log for peak_vram_gb, steps_per_sec, and final_loss."""
    raise NotImplementedError


def compute_speedup_ratio(unsloth_steps_per_sec: float, vanilla_steps_per_sec: float) -> float:
    """Compute the relative speedup ratio between Unsloth and vanilla execution."""
    raise NotImplementedError
