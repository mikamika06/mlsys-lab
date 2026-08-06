import ref
from ncu_diag.parser import parse_ncu_summary
from reference.parser import parse_ncu_summary as ref_parse


def check(workdir):
    out = {"metrics_parsed_match": 0.0}
    try:
        got_before = parse_ncu_summary(ref.BEFORE_CSV)
        want_before = ref_parse(ref.BEFORE_CSV)
        got_after = parse_ncu_summary(ref.AFTER_CSV)
        want_after = ref_parse(ref.AFTER_CSV)

        if got_before == want_before and got_after == want_after:
            out["metrics_parsed_match"] = 1.0
        else:
            out["_note"] = f"Parsed dict mismatch. Got before: {got_before}, want: {want_before}"
    except Exception as e:
        out["_note"] = f"Exception during parse: {type(e).__name__}: {str(e)[:100]}"
    return out
