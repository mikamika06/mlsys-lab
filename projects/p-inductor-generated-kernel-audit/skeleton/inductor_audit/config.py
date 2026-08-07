class CompilerConfig:
    def __init__(self, max_autotune=False, enable_fusion=True, min_fusion_size=16):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError
