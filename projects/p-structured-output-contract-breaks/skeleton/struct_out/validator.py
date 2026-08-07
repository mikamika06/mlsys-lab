class Validator:
    def __init__(self, schema):
        raise NotImplementedError()

    def validate(self, data):
        raise NotImplementedError()

class RobustPipeline:
    def __init__(self, schema):
        raise NotImplementedError()

    def run_batch(self, n):
        raise NotImplementedError()
