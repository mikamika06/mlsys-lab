import ref
from layout.derivation import derive_block_size


def check(workdir):
    out = {"derivations_matched": 0.0}
    ok = True
    for isa in ref.ISAS:
        for c in ref.CHANNELS:
            try:
                got = derive_block_size(isa, c)
            except Exception:
                ok = False
                break
            def local_derive(i, ch):
                b = 16 if i == "avx512" else (8 if i == "avx2" else (4 if i == "neon" else 1))
                while ch % b != 0 and b > 1:
                    b //= 2
                return b
            if got != local_derive(isa, c):
                ok = False
                break
        if not ok:
            break
    if ok:
        out["derivations_matched"] = 1.0
    return out
