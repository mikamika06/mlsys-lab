import ref


def check(workdir):
    from chattpl.extract import extract_chat_template

    out = {"extractions_matched": 0.0}
    tmpl = "template_string_abc_123"
    fix = ref.make_fixture(tmpl)
    got = extract_chat_template(fix)
    if got == tmpl:
        out["extractions_matched"] = 1.0
    else:
        out["_note"] = f"got {repr(got)}, want {repr(tmpl)}"
    return out
