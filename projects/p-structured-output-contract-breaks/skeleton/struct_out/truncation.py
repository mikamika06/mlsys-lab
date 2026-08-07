class TruncationHandler:
    def __init__(self, schema):
        raise NotImplementedError()

    def is_truncated(self, raw):
        raise NotImplementedError()

    def repair(self, raw):
        raise NotImplementedError()

    def validate(self, raw):
        raise NotImplementedError()
