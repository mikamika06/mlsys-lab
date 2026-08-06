class AttentionConfig:
    """Configuration for QK-normed attention with custom scale."""

    def __init__(self, head_dim: int, custom_scale: float = None, softcap: float = None, eps: float = 1e-6):
        self.head_dim = head_dim
        self.custom_scale = custom_scale
        self.softcap = softcap
        self.eps = eps

    def get_scale(self) -> float:
        if self.custom_scale is not None:
            return float(self.custom_scale)
        return float(1.0 / (self.head_dim ** 0.5))
