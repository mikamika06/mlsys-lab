import ref


def check(workdir):
    from embedopt.truncation import truncate_sequence

    out = {"truncation_policy_matched": 0.0}
    num_ctx = 16
    test_cases = [
        ([1, 2, 3], 16, "error"),
        (list(range(20)), 16, "truncate_right"),
        (list(range(20)), 16, "truncate_left"),
    ]

    ok = True
    for seq, ctx, pol in test_cases:
        want = ref.process_sequence_truncation(seq, ctx, pol)
        try:
            got = truncate_sequence(seq, ctx, pol)
            if got != want:
                ok = False
                out["_note"] = f"Mismatch for policy {pol}. Want {want}, got {got}"
                break
        except Exception as e:
            ok = False
            out["_note"] = f"Exception raised for policy {pol}: {type(e).__name__}: {str(e)}"
            break

    try:
        truncate_sequence(list(range(20)), 16, "error")
        ok = False
        out["_note"] = "Expected ValueError when sequence exceeds context under 'error' policy, but no error was raised."
    except ValueError:
        pass
    except Exception as e:
        ok = False
        out["_note"] = f"Expected ValueError, got {type(e).__name__}"

    if ok:
        out["truncation_policy_matched"] = 1.0

    return out
