import ref


def check(workdir):
    from chatparse.render import render_chat_template

    out = {"templates_matched": 0.0}
    ok = 0
    for sample in ref.TEMPLATE_SAMPLES:
        want = ref.render_template(sample["template"], sample["messages"], sample["add_generation_prompt"])
        got = render_chat_template(sample["template"], sample["messages"], sample["add_generation_prompt"])
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"got {repr(got)}, want {repr(want)}"
    out["templates_matched"] = float(ok)
    return out
