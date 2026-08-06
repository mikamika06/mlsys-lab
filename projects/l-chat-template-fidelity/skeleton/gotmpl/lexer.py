class TemplateError(Exception):
    pass


class Text:
    def __init__(self, s):
        self.s = s


class Action:
    def __init__(self, s, trim_left, trim_right):
        self.s = s
        self.trim_left = trim_left
        self.trim_right = trim_right


def tokenize(src):
    """The template split into Text and Action tokens.

    `{{- ` trims the whitespace that precedes the action and ` -}}` trims what
    follows it. A `{{/* comment */}}` produces an action that renders nothing
    but still trims. An unclosed action is a TemplateError.
    """
    raise NotImplementedError
