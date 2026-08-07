class RuntimeLimiter:
    def __init__(self, max_memory):
        raise NotImplementedError

    def check_and_apply(self, usage):
        raise NotImplementedError

def degrade_gracefully(config, memory_limit):
    raise NotImplementedError

def run_safe_mode(configs):
    raise NotImplementedError
