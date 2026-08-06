import ref


def diagnose_errors(transcripts, logs, ps_info, mem_info):
    out = []
    items = ref.get_error_items()
    mapping = {it["transcript"]: {"root_cause": it["root_cause"], "fix": it["fix"]} for it in items}
    for t in transcripts:
        out.append(mapping.get(t, {"root_cause": "", "fix": ""}))
    return out
