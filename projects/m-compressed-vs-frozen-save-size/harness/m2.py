import os
import tempfile
import ref
from quantlibs.loader import compare_load_times

def check(workdir):
    out = {"load_match": 0.0, "format_order_match": 0.0}
    with tempfile.TemporaryDirectory() as tmpdir:
        formats, paths = ref.generate_fixtures(tmpdir)
        try:
            res = compare_load_times(formats, paths)
            if isinstance(res, dict) and "times" in res and "fastest" in res:
                out["load_match"] = 1.0
                if res["fastest"] in formats:
                    out["format_order_match"] = 1.0
        except Exception:
            pass
    return out
