def check(workdir):
    from minja_tools.diff import TemplateDiff
    td = TemplateDiff("Hello {name}", "Hello {name}!")
    diff = td.render_diff({"name": "World"})
    if isinstance(diff, str) and len(diff) > 0:
        return {"diff_matched": 1.0}
    return {"diff_matched": 0.0}
