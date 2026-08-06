import sys
import os
import ref

sys.path.insert(0, os.path.abspath("."))


def check(workdir):
    from nsys_diag.osrt import diagnose_osrt_bottleneck, categorize_syscall_time
    import reference.nsys_diag.osrt as ref_osrt

    out = {"osrt_reports_matched": 0, "total_reports": 10}
    ok = 0

    for i in range(10):
        rows = ref.generate_osrt_dataset(seed=100 + i)

        want_cats = ref_osrt.categorize_syscall_time(rows)
        got_cats = categorize_syscall_time(rows)

        want_diag = ref_osrt.diagnose_osrt_bottleneck(rows)
        got_diag = diagnose_osrt_bottleneck(rows)

        cats_match = all(abs(want_cats[k] - got_cats.get(k, -1.0)) < 1e-4 for k in want_cats)
        diag_match = (
            want_diag["primary_bottleneck"] == got_diag.get("primary_bottleneck") and
            want_diag["is_io_bound"] == got_diag.get("is_io_bound") and
            abs(want_diag["io_ratio"] - got_diag.get("io_ratio", -1.0)) < 1e-4
        )

        if cats_match and diag_match:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"Report {i} mismatch: want {want_diag}, got {got_diag}"

    out["osrt_reports_matched"] = ok
    return out
