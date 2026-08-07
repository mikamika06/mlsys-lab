import ref


def check(workdir):
    from opset.drift import migrate_squeeze_11, infer_resize_shape

    out = {"migrate_matched": 0.0, "resize_matched": 0.0}

    tests_m = [
        ref.NODES_1,
        ref.NODES_2,
        [{"name": "n1", "op_type": "Add", "inputs": [], "outputs": [], "attributes": {}}]
    ]
    ok_m = 0
    for nodes in tests_m:
        want = ref.migrate_squeeze_11(nodes)
        got = migrate_squeeze_11(nodes)
        if want == got:
            ok_m += 1
        elif "_note_m" not in out:
            out["_note_m"] = f"migration mismatch on node {nodes[0]['name']}"
    out["migrate_matched"] = float(ok_m)

    tests_r = [
        ([10, 20], [1.5, 2.0], None, 10),
        ([10, 20], [1.5, 2.0], [15, 40], 11),
        ([10, 20], [1.5, 2.0], [], 11),
        ([10, 20], [2.0, 2.0], None, 11),
    ]
    ok_r = 0
    for shape, scales, sizes, opset in tests_r:
        want = ref.infer_resize_shape(shape, scales, sizes, opset)
        got = infer_resize_shape(shape, scales, sizes, opset)
        if want == got:
            ok_r += 1
        elif "_note_r" not in out:
            out["_note_r"] = f"resize mismatch on opset {opset}"
    out["resize_matched"] = float(ok_r)

    return out
