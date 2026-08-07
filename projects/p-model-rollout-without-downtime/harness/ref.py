def get_reference_manager():
    from rollout.manager import ModelManager
    m = ModelManager()
    m.load_version("v1", lambda x: x * 2)
    m.load_version("v2", lambda x: x * 3)
    return m


def get_reference_policy():
    from rollout.policy import RolloutPolicy
    return RolloutPolicy([0.1, 0.5, 1.0], 0.05)
