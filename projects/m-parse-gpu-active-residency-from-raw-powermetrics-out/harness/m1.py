import sys

sys.path.insert(0, ".")
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from profile_parser.parser import parse_gpu_active_residency

    text, want_gpu, _ = ref.generate_powermetrics_fixture()
    got_gpu = parse_gpu_active_residency(text)

    out = {
        "samples_parsed_matched": 0.0,
        "residency_rel_err": 1.0,
    }

    if len(got_gpu) != len(want_gpu):
        out["_note"] = f"Expected {len(want_gpu)} samples, got {len(got_gpu)}"
        return out

    out["samples_parsed_matched"] = 1.0

    errs = [abs(g - w) / (abs(w) + 1e-9) for g, w in zip(got_gpu, want_gpu)]
    max_err = max(errs) if errs else 0.0
    out["residency_rel_err"] = float(max_err)

    return out
