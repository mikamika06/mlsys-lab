class DataLoaderPipeline:
    def __init__(self, pin_memory=True, prefetch_factor=2):
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

    def process(self, batch):
        if self.pin_memory:
            return [x.pin_memory() if hasattr(x, "pin_memory") else x for x in batch]
        return batch
