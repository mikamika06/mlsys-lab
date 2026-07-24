import sys


def model_split_dict_savings(m: int) -> dict:
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

    combined_dicts = []
    for i in range(m):
        combined_dicts.append({"x": i, "y": i + 1, "z": i + 2})

    combined_bytes = sum(sys.getsizeof(d) for d in combined_dicts)

    class SlotModel:
        __slots__ = ("x", "y", "z")

    slot_instances = []
    for i in range(m):
        obj = SlotModel()
        obj.x = i
        obj.y = i + 1
        obj.z = i + 2
        slot_instances.append(obj)

    slots_bytes = sum(sys.getsizeof(obj) for obj in slot_instances)

    return {
        "split_bytes": split_bytes,
        "combined_bytes": combined_bytes,
        "slots_bytes": slots_bytes,
        "savings_ratio": combined_bytes / split_bytes,
    }
