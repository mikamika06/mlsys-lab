import sys
import numpy as np


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from router.bakeoff import run_bakeoff, compute_p95_ttft
        from router.tuning import grid_search_alpha
        import ref

        errs = []

        for trace in ref.TRACES:
            got_bakeoff = run_bakeoff(
                trace,
                num_workers=4,
                max_blocks_per_worker=16,
                block_size=4,
                prefill_rate=1000.0,
                decode_rate=100.0,
                alpha=0.5,
            )
            ref_bakeoff = ref.run_bakeoff(
                trace,
                num_workers=4,
                max_blocks_per_worker=16,
                block_size=4,
                prefill_rate=1000.0,
                decode_rate=100.0,
                alpha=0.5,
            )

            for pol in ["rr", "prefix", "kv_aware"]:
                res_got = got_bakeoff[pol]
                res_ref = ref_bakeoff[pol]
                if len(res_got) != len(res_ref):
                    errs.append(1.0)
                    continue

                for r_g, r_r in zip(res_got, res_ref):
                    if r_g["worker_id"] != r_r["worker_id"]:
                        errs.append(0.5)
                    ttft_diff = abs(r_g["ttft"] - r_r["ttft"])
                    errs.append(ttft_diff / (abs(r_r["ttft"]) + 1e-6))

            got_alpha, _ = grid_search_alpha(
                trace,
                num_workers=4,
                max_blocks_per_worker=16,
                block_size=4,
                prefill_rate=1000.0,
                decode_rate=100.0,
            )
            ref_alpha, _ = ref.grid_search_alpha(
                trace,
                num_workers=4,
                max_blocks_per_worker=16,
                block_size=4,
                prefill_rate=1000.0,
                decode_rate=100.0,
            )
            if abs(got_alpha - ref_alpha) > 1e-4:
                errs.append(1.0)
            else:
                errs.append(0.0)

        rel_err = float(np.mean(errs)) if errs else 1.0
        return {"rel_err": rel_err}
    except Exception:
        return {"rel_err": 1.0}
    finally:
        if workdir in sys.path:
            sys.path.remove(workdir)
