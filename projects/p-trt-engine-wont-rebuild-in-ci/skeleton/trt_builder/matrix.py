class PortabilityMatrix:
    def __init__(self):
        raise NotImplementedError

    def add_entry(self, source_env, target_env, compatible, reason=""):
        raise NotImplementedError

    def can_deploy(self, source_env, target_env):
        raise NotImplementedError

    def generate_report(self):
        raise NotImplementedError
