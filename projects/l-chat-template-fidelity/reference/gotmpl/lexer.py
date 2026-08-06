class TemplateError(Exception):
    pass


class Text:
    def __init__(self, s):
        self.s = s

    def __repr__(self):
        return "Text(%r)" % self.s


class Action:
    def __init__(self, s, trim_left, trim_right):
        self.s = s
        self.trim_left = trim_left
        self.trim_right = trim_right

    def __repr__(self):
        return "Action(%r,%s,%s)" % (self.s, self.trim_left, self.trim_right)


SPACE = " \t\r\n"


def tokenize(src):
    out = []
    i = 0
    n = len(src)
    while i < n:
        j = src.find("{{", i)
        if j < 0:
            out.append(Text(src[i:]))
            break
        if j > i:
            out.append(Text(src[i:j]))
        k = src.find("}}", j)
        if k < 0:
            raise TemplateError("unclosed action at byte %d" % j)
        body = src[j + 2:k]
        trim_left = body.startswith("-") and (len(body) == 1 or body[1] in SPACE)
        if trim_left:
            body = body[1:]
        trim_right = body.endswith("-") and (len(body) == 1 or body[-2] in SPACE)
        if trim_right:
            body = body[:-1]
        body = body.strip()
        if body.startswith("/*"):
            if not body.endswith("*/"):
                raise TemplateError("unclosed comment at byte %d" % j)
            body = ""
            out.append(Action("", trim_left, trim_right))
        else:
            if not body:
                raise TemplateError("empty action at byte %d" % j)
            out.append(Action(body, trim_left, trim_right))
        i = k + 2

    for idx, tok in enumerate(out):
        if not isinstance(tok, Action):
            continue
        if tok.trim_left and idx and isinstance(out[idx - 1], Text):
            out[idx - 1].s = out[idx - 1].s.rstrip(SPACE)
        if tok.trim_right and idx + 1 < len(out) and isinstance(out[idx + 1], Text):
            out[idx + 1].s = out[idx + 1].s.lstrip(SPACE)
    return out
