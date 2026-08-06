import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    from mpsgraph.mapping import map_recorded_sequence

    out = {"mapping_accuracy": 0.0}
    total = float(len(ref.TEST_OP_SEQUENCES))
    correct = 0

    for seq in ref.TEST_OP_SEQUENCES:
        want = ref.ref_map_recorded_sequence(seq)
        try:
            got = map_recorded_sequence(seq)
            if got == want:
                correct += 1
            elif "_note" not in out:
                out["_note"] = f"seq {seq}: expected {want}, got {got}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"seq {seq} raised {type(e).__name__}: {e}"

    out["mapping_accuracy"] = float(correct) / total if total > 0 else 0.0
    return out
