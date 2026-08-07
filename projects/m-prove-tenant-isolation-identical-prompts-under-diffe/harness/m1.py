import sys
import ref

sys.path.insert(0, ".")


def check(workdir):
    out = {"isolation_matched": 0.0, "zero_shared_blocks": 0.0}
    try:
        from vllm_sec.isolation import compute_block_hashes, check_tenant_isolation
    except ImportError as e:
        out["_note"] = f"ImportError: {e}"
        return out

    try:
        ref_isolated, ref_shared = ref.check_tenant_isolation(ref.REQUESTS, ref.BLOCK_SIZE)
        got_isolated, got_shared = check_tenant_isolation(ref.REQUESTS, ref.BLOCK_SIZE)

        h1 = compute_block_hashes(ref.REQUESTS[0]["tokens"], ref.BLOCK_SIZE, ref.REQUESTS[0]["tenant_salt"])
        ref_h1 = ref.compute_block_hashes(ref.REQUESTS[0]["tokens"], ref.BLOCK_SIZE, ref.REQUESTS[0]["tenant_salt"])

        if h1 == ref_h1 and got_isolated == ref_isolated and got_shared == ref_shared:
            out["isolation_matched"] = 1.0

        if got_shared == 0 and got_isolated is True:
            out["zero_shared_blocks"] = 1.0
        else:
            out["_note"] = f"Expected 0 shared blocks and True isolation, got shared={got_shared}, isolated={got_isolated}"
    except Exception as e:
        out["_note"] = f"Execution failed: {type(e).__name__}: {e}"

    return out
