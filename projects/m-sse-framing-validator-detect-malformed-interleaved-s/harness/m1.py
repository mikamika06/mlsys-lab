import ref
import sys
import os

sys.path.insert(0, os.getcwd())


def check(workdir):
    from ssevall.validator import validate_sse_stream, FramingError

    out = {"streams_checked": 0.0}
    passed = 0
    for raw in ref.VALID_STREAMS:
        try:
            res = validate_sse_stream(raw)
            if isinstance(res, list) and len(res) > 0:
                passed += 1
        except Exception:
            pass

    for raw in ref.MALFORMED_STREAMS:
        try:
            validate_sse_stream(raw)
        except FramingError:
            passed += 1
        except Exception:
            pass

    out["streams_checked"] = float(passed)
    return out
