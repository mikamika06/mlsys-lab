import ref


def check(workdir):
    from irhist.parser import parse_ir_xml
    from irhist.histogram import build_histogram

    out = {"histogram_matched": 0.0}
    ok = 0
    for xml in ref.NETS:
        ops_ref = parse_ir_xml(xml)
        hist_ref = build_histogram(ops_ref)

        try:
            from irhist import parser, histogram
            ops_got = parser.parse_ir_xml(xml)
            hist_got = histogram.build_histogram(ops_got)
            if hist_got == hist_ref:
                ok += 1
        except Exception:
            pass

    out["histogram_matched"] = float(ok)
    return out
