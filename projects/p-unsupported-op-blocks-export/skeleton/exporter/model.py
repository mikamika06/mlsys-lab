class ModelExporter:
    def __init__(self, config):
        raise NotImplementedError

    def localize_unsupported(self):
        raise NotImplementedError

    def export_full(self):
        raise NotImplementedError
