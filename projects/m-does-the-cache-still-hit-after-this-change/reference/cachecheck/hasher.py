import hashlib
import json
import numpy as np


def stable_hash(obj):
    def _serialize(o):
        if isinstance(o, (int, float, str, bool, type(None))):
            return (type(o).__name__, o)
        elif isinstance(o, (list, tuple)):
            return [_serialize(x) for x in o]
        elif isinstance(o, dict):
            sorted_items = sorted(o.items(), key=lambda kv: str(kv[0]))
            return {str(k): _serialize(v) for k, v in sorted_items}
        elif isinstance(o, np.ndarray):
            return ("ndarray", o.dtype.str, o.shape, o.tobytes())
        else:
            return ("str", str(o))
    serialized = _serialize(obj)
    dumped = json.dumps(serialized, sort_keys=True, default=str)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()
