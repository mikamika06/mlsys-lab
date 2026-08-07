import ref


def check(workdir):
    from gradscaler.sim import identify_skipped_steps, next_doubling_step

    out = {"skipped_matched": 0.0, "doubling_matched": 0.0, "total": float(len(ref.FIXTURES))}
    skipped_ok = 0
    doubling_ok = 0

    for i, seq in enumerate(ref.FIXTURES):
        try:
            got_skipped = identify_skipped_steps(seq)
            if ref.identify_skipped_steps(seq) == got_skipped:
                skipped_ok += 1

            got_doubling = next_doubling_step(seq)
            if ref.next_doubling_step(seq) == got_doubling:
                doubling_ok += 1
        except Exception:
            pass

    out["skipped_matched"] = float(skipped_ok)
    out["doubling_matched"] = float(doubling_ok)
    return out
