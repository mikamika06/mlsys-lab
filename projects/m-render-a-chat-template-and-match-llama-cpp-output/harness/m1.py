import ref


def check(workdir):
    from chattpl.render import render_chat

    out = {"renders_matched": 0.0}
    template = "<|start|>{role}\n{content}<|end|>\n"
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ]
    want = ref.render_chat(template, messages)
    got = render_chat(template, messages)
    if got == want:
        out["renders_matched"] = 1.0
    else:
        out["_note"] = f"got {repr(got)}, want {repr(want)}"
    return out
