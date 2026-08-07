import ref


def check(workdir):
    from qindex.writer import build_shard_index, serialize_index
    from qindex.validation import validate_index_structure

    out = {"bytes_correct": 0.0, "format_valid": 0.0}
    index_data = ref.build_index(ref.CHECKPOINTS)
    serialized = serialize_index(index_data)

    try:
        parsed = validate_index_structure(serialized)
        if parsed:
            out["format_valid"] = 1.0
    except Exception as e:
        out["_note"] = f"validation failed: {e}"
        return out

    got_index = build_shard_index(ref.CHECKPOINTS)
    if got_index == index_data:
        out["bytes_correct"] = 1.0
    else:
        out["_note"] = "serialized index content does not match oracle structure"

    return out
