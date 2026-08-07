class CompilerConfig:
    def __init__(self, max_autotune=False, enable_fusion=True, min_fusion_size=16):
        self.max_autotune = max_autotune
        self.enable_fusion = enable_fusion
        self.min_fusion_size = min_fusion_size

    def to_dict(self):
        return {
            "max_autotune": self.max_autotune,
            "enable_fusion": self.enable_fusion,
            "min_fusion_size": self.min_fusion_size,
        }
