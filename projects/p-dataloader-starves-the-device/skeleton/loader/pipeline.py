class DataLoaderPipeline:
    def __init__(self, pin_memory=True, prefetch_factor=2):
        raise NotImplementedError

    def process(self, batch):
        raise NotImplementedError
