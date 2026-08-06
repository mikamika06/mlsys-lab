import ref

def check(workdir):
    from tflite_tools.diff import structural_diff

    t1 = [{"name": "node_a", "scale": 0.1, "zero_point": 0}, {"name": "node_b", "scale": 0.2, "zero_point": 0}]
    t2 = [{"name": "node_a", "scale": 0.1, "zero_point": 0}, {"name": "node_b_renamed", "scale": 0.2, "zero_point": 0}]

    fb1 = ref.generate_mock_flatbuffer(t1)
    fb2 = ref.generate_mock_flatbuffer(t2)

    diff = structural_diff(fb1, fb2)
    matched = 1 if isinstance(diff, dict) and len(diff.get("changed_tensors", [])) > 0 else 0
    return {"diff_match": float(matched)}
