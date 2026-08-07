import ref
import numpy as np


def check(workdir):
    from nf4.dequant import get_nf4_table, unpack_4bit

    out = {"table_max_diff": 1.0, "unpack_matched": 0.0}

    try:
        table = get_nf4_table()
        ref_table = ref.get_nf4_table()
        out["table_max_diff"] = float(np.max(np.abs(table - ref_table)))
    except Exception as e:
        out["_note_table"] = f"table error: {type(e).__name__}: {str(e)}"

    try:
        packed = np.array([0x41, 0x9A, 0xFF, 0x00], dtype=np.uint8)
        got = unpack_4bit(packed)
        want = ref.unpack_4bit(packed)
        if np.array_equal(got, want):
            out["unpack_matched"] = 1.0
        else:
            out["_note_unpack"] = f"unpack mismatch. got {got[:4]}, want {want[:4]}"
    except Exception as e:
        out["_note_unpack"] = f"unpack error: {type(e).__name__}: {str(e)}"

    return out
