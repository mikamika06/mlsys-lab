class FailureCollector:
    def __init__(self, schema):
        raise NotImplementedError()

    def classify(self, raw):
        raise NotImplementedError()

def collect_and_classify(corpus, schema):
    raise NotImplementedError()
