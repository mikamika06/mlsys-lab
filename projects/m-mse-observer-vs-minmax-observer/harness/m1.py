import ref


def check(workdir):
    from quant.scheme import parse_scheme

    out = {"schemes_matched": 0.0}
    ok = 0
    for name in ref.SCHEMES:
        args = parse_scheme(name)
        ref_args = __import__("reference.quant.scheme", fromlist=["parse_scheme"]).parse_scheme(name)
        if args.bits == ref_args.bits and args.symmetric == ref_args.symmetric and args.granularity == ref_args.granularity:
            ok += 1
    out["schemes_matched"] = float(ok)
    return out
