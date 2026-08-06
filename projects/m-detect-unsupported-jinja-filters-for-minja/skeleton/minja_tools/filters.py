class UnsupportedFilterDetector:
    def __init__(self, supported_filters):
        raise NotImplementedError

    def find_unsupported(self, template_str):
        raise NotImplementedError
