class UnsupportedFilterDetector:
    def __init__(self, supported_filters):
        self.supported_filters = set(supported_filters)

    def find_unsupported(self, template_str):
        import re
        pattern = r"\|\s*([a-zA-Z_][a-zA-Z0-9_]*)"
        found = re.findall(pattern, template_str)
        unsupported = sorted(list(set(f for f in found if f not in self.supported_filters)))
        return unsupported
