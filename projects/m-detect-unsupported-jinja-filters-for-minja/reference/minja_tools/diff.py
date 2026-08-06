class TemplateDiff:
    def __init__(self, template_a, template_b):
        self.template_a = template_a
        self.template_b = template_b

    def render_diff(self, context):
        from difflib import ndiff
        render_a = self.render(self.template_a, context)
        render_b = self.render(self.template_b, context)
        diff = list(ndiff(render_a.splitlines(keepends=True), render_b.splitlines(keepends=True)))
        return "".join(diff)

    def render(self, template, context):
        return template.format(**context)
