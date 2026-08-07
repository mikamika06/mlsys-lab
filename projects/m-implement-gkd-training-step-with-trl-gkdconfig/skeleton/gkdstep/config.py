from dataclasses import dataclass


@dataclass
class GKDConfig:
    """Hyperparameter configuration for GKD training steps."""

    temperature: float = 1.0
    lmbda: float = 0.5
    beta: float = 0.5
    divergence_type: str = "forward_kl"
    max_new_tokens: int = 16
