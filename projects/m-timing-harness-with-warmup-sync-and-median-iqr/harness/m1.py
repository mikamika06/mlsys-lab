import sys
import time
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        import benchmark
        import torch
        import numpy as np

        out = {"runs_warmup": 0.0, "syncs_cuda": 0.0, "correct_stats": 0.0}

        calls = [0]

        def dummy_fn():
            calls[0] += 1

        try:
            benchmark.benchmark_step(dummy_fn, is_cuda=False, warmup=5, reps=10)
            if calls[0] == 15:
                out["runs_warmup"] = 1.0
        except NotImplementedError:
            pass
        except Exception as e:
            out["_note"] = f"failed on basic run: {e}"

        sync_calls = [0]
        original_sync = torch.cuda.synchronize

        def mock_sync():
            sync_calls[0] += 1

        torch.cuda.synchronize = mock_sync
        calls[0] = 0

        try:
            benchmark.benchmark_step(dummy_fn, is_cuda=True, warmup=2, reps=5)
            if sync_calls[0] >= 5:
                out["syncs_cuda"] = 1.0
        except NotImplementedError:
            pass
        except Exception:
            pass
        finally:
            torch.cuda.synchronize = original_sync

        try:
            res = benchmark.benchmark_step(lambda: None, is_cuda=False, warmup=0, reps=4)
            if "times" in res and "median" in res and "iqr" in res:
                med, iqr = ref.get_percentiles(res["times"])
                if np.isclose(res["median"], med) and np.isclose(res["iqr"], iqr):
                    out["correct_stats"] = 1.0
        except Exception:
            pass

        return out
    finally:
        sys.path.pop(0)
