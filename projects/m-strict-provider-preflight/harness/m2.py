import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {"node_counts_matched": 0.0, "distribution_matched": 0.0}
    try:
        from ortpreflight.logs import (
            analyze_ep_distribution,
            parse_ep_node_counts,
        )
    except Exception as e:
        out["_note"] = f"failed to import learner module: {e}"
        return out

    ok_counts = 0
    for i, log_text in enumerate(ref.LOG_CASES):
        want = ref.parse_ep_node_counts(log_text)
        try:
            got = parse_ep_node_counts(log_text)
            if got == want:
                ok_counts += 1
            elif "_note" not in out:
                out["_note"] = f"log counts case {i}: got {got}, reference {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"log counts case {i} raised {e}"
    out["node_counts_matched"] = float(ok_counts)

    ok_dist = 0
    for i, (log_text, target_ep) in enumerate(ref.LOG_DISTRIBUTION_CASES):
        want = ref.analyze_ep_distribution(log_text, target_ep)
        try:
            got = analyze_ep_distribution(log_text, target_ep)
            if got == want:
                ok_dist += 1
            elif "_note" not in out:
                out["_note"] = (
                    f"log distribution case {i}: got {got}, reference {want}"
                )
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"log distribution case {i} raised {e}"
    out["distribution_matched"] = float(ok_dist)
    return out
