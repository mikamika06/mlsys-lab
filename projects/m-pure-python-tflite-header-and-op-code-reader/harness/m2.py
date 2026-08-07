import ref


def check(workdir):
    from tflite_tools.parser import attribute_bytes
    from tflite_tools.stripper import strip_weights

    out = {"attribution_match": 0.0, "stripped_size_match": 0.0}
    attr_ok = 0
    strip_ok = 0

    for i, m_bytes in enumerate(ref.MODELS):
        want_attr = attribute_bytes(m_bytes)
        stripped_bytes = strip_weights(m_bytes)
        want_stripped_attr = attribute_bytes(stripped_bytes)

        try:
            got_attr = attribute_bytes(m_bytes)
            got_stripped = strip_weights(m_bytes)
            got_stripped_attr = attribute_bytes(got_stripped)
        except Exception as e:
            out["_note"] = f"model {i} error: {type(e).__name__}"
            return out

        if got_attr == want_attr:
            attr_ok += 1
        if len(got_stripped) == len(m_bytes) and got_stripped_attr["buffer_bytes"] <= want_stripped_attr["buffer_bytes"]:
            strip_ok += 1

    out["attribution_match"] = 1.0 if attr_ok == len(ref.MODELS) else 0.0
    out["stripped_size_match"] = 1.0 if strip_ok == len(ref.MODELS) else 0.0
    return out
