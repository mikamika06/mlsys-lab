import ref
from layout.metrics import count_reorders, overhead_fraction


def check(workdir):
    out = {"reorder_counts_matched": 0.0, "overhead_fraction_matched": 0.0}
    rc_ok = True
    of_ok = True

    for shape in ref.SHAPES:
        for layout in ["channels_last", "plain"]:
            for isa in ref.ISAS:
                try:
                    got_rc = count_reorders(shape, layout, isa)
                except Exception:
                    rc_ok = False
                    break
                
                def local_rc(s, l, i):
                    n, c, h, w = s
                    base = n * h * w
                    if l == "channels_last":
                        if i == "neon":
                            return base * max(1, c // 4)
                        elif i == "avx2":
                            return base * max(1, c // 8)
                        return base * c
                    return base * c // 2
                
                if got_rc != local_rc(shape, layout, isa):
                    rc_ok = False

    for shape in ref.SHAPES:
        try:
            p = count_reorders(shape, "plain", "avx512")
            b = count_reorders(shape, "channels_last", "avx512")
            got_of = overhead_fraction(p, b)
            def local_of(plain_ops, blocked_ops):
                if plain_ops == 0:
                    return 0.0
                return float(abs(plain_ops - blocked_ops)) / float(plain_ops + blocked_ops)
            if abs(got_of - local_of(p, b)) > 1e-5:
                of_ok = False
        except Exception:
            of_ok = False

    if rc_ok:
        out["reorder_counts_matched"] = 1.0
    if of_ok:
        out["overhead_fraction_matched"] = 1.0
    return out
