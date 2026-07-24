import math
import sys


def _oracle(m):
    class SplitModel:
        pass

    split_instances = []
    for i in range(m):
        obj = SplitModel()
        obj.x = i
        obj.y = i + 1
        obj.z = i + 2
        split_instances.append(obj)

    split_bytes = sum(sys.getsizeof(obj.__dict__) for obj in split_instances)

    combined = []
    for i in range(m):
        combined.append({"x": i, "y": i + 1, "z": i + 2})
    combined_bytes = sum(sys.getsizeof(d) for d in combined)

    class SlotModel:
        __slots__ = ("x", "y", "z")

    slots = []
    for i in range(m):
        obj = SlotModel()
        obj.x = i
        obj.y = i + 1
        obj.z = i + 2
        slots.append(obj)

    slots_bytes = sum(sys.getsizeof(obj) for obj in slots)

    return {
        "split_bytes": split_bytes,
        "combined_bytes": combined_bytes,
        "slots_bytes": slots_bytes,
        "savings_ratio": combined_bytes / split_bytes,
    }


def grade(sol, fx) -> dict:
    ok = 1.0
    for m in [1, 10, 100]:
        try:
            got = sol.model_split_dict_savings(m)
            ref = _oracle(m)
            if not isinstance(got, dict):
                ok = 0.0
                break
            if not math.isclose(
                float(got["savings_ratio"]),
                float(ref["savings_ratio"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break
    return {"size_ratio": ok}
