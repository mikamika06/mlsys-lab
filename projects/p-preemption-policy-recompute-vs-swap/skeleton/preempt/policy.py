class PreemptionPolicy:
    def __init__(self, config):
        raise NotImplementedError

    def decide(self, request, cluster_state):
        raise NotImplementedError
