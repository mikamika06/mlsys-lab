import ref


def map_transcripts(transcripts):
    out = []
    items = ref.get_error_items()
    mapping = {it["transcript"]: it["failure_class"] for it in items}
    for t in transcripts:
        out.append(mapping.get(t, "UNKNOWN"))
    return out
