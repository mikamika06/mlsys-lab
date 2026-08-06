import sys
sys.path.insert(0, ".")
import ref


def check(workdir):
    out = {"configs_matched": 0.0}
    try:
        from dispatch.selector import dispatch_kernel, is_eligible
    except Exception as e:
        out["_note"] = f"Import error: {e}"
        return out

    total = len(ref.CHECKPOINTS)
    matched = 0

    for ckpt in ref.CHECKPOINTS:
        ref_dispatched = ref.dispatch_kernel(ref.KERNELS, ckpt)
        try:
            got_dispatched = dispatch_kernel(ref.KERNELS, ckpt)
        except Exception as e:
            out["_note"] = f"dispatch_kernel raised: {e}"
            return out

        if ref_dispatched == got_dispatched:
            matched += 1

    if matched == total:
        out["configs_matched"] = 1.0
    else:
        out["_note"] = f"Matched {matched}/{total} checkpoint dispatches"

    return out
