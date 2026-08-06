class FallbackDiagnostics:
    def __init__(self):
        raise NotImplementedError

    def diagnose_fallback(self, kernel_config):
        raise NotImplementedError
