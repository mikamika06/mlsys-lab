class ToolDecisionEngine:
    def __init__(self, supported_filters):
        from minja_tools.filters import UnsupportedFilterDetector
        self.detector = UnsupportedFilterDetector(supported_filters)

    def needs_jinja(self, template_str, tools_present):
        unsupported = self.detector.find_unsupported(template_str)
        if unsupported and tools_present:
            return True
        return False
