import numpy as np
import ref


def check(workdir):
    out = {"bucket_assignment_match": 0.0, "padded_tensors_match": 0.0}

    try:
        from padder.bucket import assign_bucket, pad_batch
    except ImportError as e:
        out["_note"] = f"Import error: {e}"
        return out

    boundaries = [32, 64, 128, 256]
    test_lengths = [1, 15, 32, 33, 64, 100, 256]
    assign_ok = True

    for l in test_lengths:
        want = ref.ref_assign_bucket(l, boundaries)
        got = assign_bucket(l, boundaries)
        if want != got:
            assign_ok = False
            out["_note"] = f"assign_bucket({l}) got {got}, expected {want}"
            break

    if assign_ok:
        out["bucket_assignment_match"] = 1.0

    rng = np.random.RandomState(123)
    seqs = [rng.randint(1, 1000, size=rng.randint(5, 120)).tolist() for _ in range(8)]
    want_pad, want_mask, want_b = ref.ref_pad_batch(seqs, boundaries, pad_val=-1)

    try:
        got_pad, got_mask, got_b = pad_batch(seqs, boundaries, pad_val=-1)
        if (
            got_b == want_b
            and np.array_equal(got_pad, want_pad)
            and np.array_equal(got_mask, want_mask)
        ):
            out["padded_tensors_match"] = 1.0
        else:
            out["_note"] = f"pad_batch output mismatch. bucket: got {got_b}, want {want_b}"
    except Exception as e:  # noqa: BLE001
        out["_note"] = f"pad_batch exception: {e}"

    return out
