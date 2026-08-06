"""Console log parser and metrics computation."""
import re


def parse_unsloth_log(log_text: str) -> dict:
    """Parse Unsloth console log for peak_vram_gb, steps_per_sec, and final_loss."""
    vram_match = re.search(r"(?:Peak memory reserved|Peak VRAM allocated|Peak VRAM):\s*([\d\.]+)\s*GB", log_text, re.IGNORECASE)
    vram = float(vram_match.group(1)) if vram_match else None

    speed_match = re.search(r"(\d+(?:\.\d+)?)\s*steps/s", log_text, re.IGNORECASE)
    if not speed_match:
        s_per_step = re.search(r"(\d+(?:\.\d+)?)\s*s/step", log_text, re.IGNORECASE)
        speed = 1.0 / float(s_per_step.group(1)) if s_per_step and float(s_per_step.group(1)) > 0 else None
    else:
        speed = float(speed_match.group(1))

    loss_matches = re.findall(r"'loss':\s*([\d\.]+)", log_text)
    if not loss_matches:
        loss_matches = re.findall(r"loss\s*=\s*([\d\.]+)", log_text, re.IGNORECASE)
    loss = float(loss_matches[-1]) if loss_matches else None

    return {
        "peak_vram_gb": round(vram, 4) if vram is not None else None,
        "steps_per_sec": round(speed, 4) if speed is not None else None,
        "final_loss": round(loss, 4) if loss is not None else None,
    }


def compute_speedup_ratio(unsloth_steps_per_sec: float, vanilla_steps_per_sec: float) -> float:
    """Compute the relative speedup ratio between Unsloth and vanilla execution."""
    if vanilla_steps_per_sec <= 0.0:
        raise ValueError("Vanilla throughput must be positive")
    return round(unsloth_steps_per_sec / vanilla_steps_per_sec, 4)
