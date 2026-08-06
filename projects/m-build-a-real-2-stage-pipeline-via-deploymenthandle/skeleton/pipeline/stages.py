class StageOne:
    def __init__(self, handle=None):
        raise NotImplementedError

    def process(self, data):
        raise NotImplementedError


class StageTwo:
    def process(self, data):
        raise NotImplementedError


class PipelineOrchestrator:
    def __init__(self, stage_one, stage_two):
        raise NotImplementedError

    def run(self, payload):
        raise NotImplementedError
