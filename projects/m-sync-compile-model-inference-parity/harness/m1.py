import ref


def check(workdir):
    from ovruntime.core import Core

    out = {"rel_err": 1.0, "parity_matched": 0.0}
    max_err = 0.0
    matched = True

    try:
        core = Core()
        for cfg in ref.TEST_CONFIGS:
            compiled = core.compile_model(cfg)
            req = compiled.create_infer_request()
            for inp in ref.TEST_INPUTS:
                ref_out = ref.reference_infer(cfg, inp)

                got_call = compiled(inp)
                err_call = ref.compute_rel_err(got_call, ref_out)

                got_req = req.infer(inp)
                err_req = ref.compute_rel_err(got_req, ref_out)

                err_parity = ref.compute_rel_err(got_call, got_req)

                cur_err = max(err_call, err_req, err_parity)
                if cur_err > max_err:
                    max_err = cur_err

                if cur_err > 1e-5:
                    matched = False

        out["rel_err"] = float(max_err)
        out["parity_matched"] = 1.0 if matched else 0.0
    except Exception as e:
        out["_note"] = f"m1 check failed: {type(e).__name__}: {str(e)[:120]}"

    return out
