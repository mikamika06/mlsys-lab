class TemplateDiff:
    def __init__(self, template_a, template_b):
        raise NotImplementedError

    def render_diff(self, context):
        raise NotImplementedError
