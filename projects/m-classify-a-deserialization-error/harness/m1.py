import struct
import ref


def check(workdir):
    from plan import parse_header

    ok = 0
    for fix in ref.FIXTURES:
        if len(fix["engine"]) >= 20:
            try:
                got = parse_header(fix["engine"])
                m, t, s, c, o = struct.unpack("<4sIIII", fix["engine"][:20])
                want = {"magic": m, "trt_version": t, "build_sm": s, "hw_compat": c, "os_id": o}
                if got == want:
                    ok += 1
            except Exception:
                pass

    return {"headers_parsed": float(ok)}
