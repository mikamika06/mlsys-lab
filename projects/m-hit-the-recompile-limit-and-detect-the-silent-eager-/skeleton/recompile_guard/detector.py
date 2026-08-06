class RecompileDetector:
    def __init__(self, limit=8):
        raise NotImplementedError

    def register(self, fn_name):
        raise NotImplementedError

    def record_compile(self, fn_name):
        raise NotImplementedError

    def record_execution(self, fn_name):
        raise NotImplementedError

    def is_limit_exceeded(self, fn_name):
        raise NotImplementedError

    def is_silent_fallback(self, fn_name):
        raise NotImplementedError

    def get_stats(self, fn_name):
        raise NotImplementedError
