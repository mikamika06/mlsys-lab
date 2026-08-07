import ref


def check(workdir):
    from ot.validator import validate_regex

    out = {"specificity_match": 0.0, "coverage_match": 0.0}
    correct_regex = "^blk\\.\\d+\\.ffn_(gate|up|down)_expts\\..*"
    bad_regex = ".*"

    if validate_regex(correct_regex):
        out["specificity_match"] = 1.0

    if not validate_regex(bad_regex):
        out["coverage_match"] = 1.0

    return out
