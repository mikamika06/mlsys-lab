import ref


def check(workdir):
    from deploy.docker import emit_docker_run

    out = {"commands_matched": 0.0, "specs": float(len(ref.SPECS))}
    ok = 0
    for i, spec in enumerate(ref.SPECS):
        want = ref.emit_docker_run(spec) if hasattr(ref, "emit_docker_run") else ""
        try:
            got = emit_docker_run(spec)
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"spec {i} raised error: {type(e).__name__}: {str(e)[:100]}"
            continue

        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"spec {i}: got {repr(got)}, want {repr(want)}"

    out["commands_matched"] = float(ok)
    return out
