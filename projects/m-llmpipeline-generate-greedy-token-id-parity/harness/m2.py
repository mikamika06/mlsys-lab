import sys
import ref

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import inference

    out = {"chat_ok": 0.0, "fallback_ok": 0.0}
    messages = [{"role": "user", "content": "Hello"}, {"role": "user", "content": "World"}]

    pipe_good = ref.MockPipeline(ref.MockModel(), has_chat=True)
    res_good = inference.format_chat_safe(pipe_good, messages)
    if res_good == "[CHAT] Hello | World":
        out["chat_ok"] = 1.0

    pipe_bad = ref.MockPipeline(ref.MockModel(), has_chat=False)
    res_bad = inference.format_chat_safe(pipe_bad, messages)
    if res_bad == "Hello\nWorld":
        out["fallback_ok"] = 1.0
    else:
        out["_note"] = f"Fallback mismatch: got {res_bad}"

    return out
