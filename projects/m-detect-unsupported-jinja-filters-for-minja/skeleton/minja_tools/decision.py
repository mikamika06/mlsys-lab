class ToolDecisionEngine:
    def __init__(self, supported_filters):
        raise NotImplementedError

    def needs_jinja(self, template_str, tools_present):
        raise NotImplementedError
