import sys
import os
import ref

def check(workdir):
    for k in list(sys.modules.keys()):
        if k == "aot_compare" or k.startswith("aot_compare."):
            del sys.modules[k]

    sys.path.insert(0, workdir)

    out = {
        "op_diff_matched": 0.0,
        "parse_ops_matched": 0.0
    }

    try:
        from aot_compare.stablehlo_diff import parse_stablehlo_op_counts, diff_stablehlo_ops

        sample_ir = ' %1 = "stablehlo.dot"(%0) \n %2 = "stablehlo.add"(%1) \n %3 = "stablehlo.add"(%2)'
        got_parse = parse_stablehlo_op_counts(sample_ir)
        want_parse = ref.parse_stablehlo_op_counts(sample_ir)

        parse_ok = 1.0 if got_parse == want_parse else 0.0

        def transform_b(ops, val):
            if val:
                return ["dot_general", "add"]
            return ops

        mock_fn = ref.MockJitFunction(
            base_ops=["dot", "add", "add"],
            flag_transforms={"fuse": transform_b}
        )

        flags_a = {"fuse": False}
        flags_b = {"fuse": True}

        got_diff = diff_stablehlo_ops(mock_fn, (1.0,), flags_a, flags_b)
        want_diff = ref.diff_stablehlo_ops(mock_fn, (1.0,), flags_a, flags_b)

        diff_ok = 1.0 if got_diff == want_diff else 0.0

        out["op_diff_matched"] = diff_ok
        out["parse_ops_matched"] = parse_ok

        if not parse_ok:
            out["_note"] = f"parse mismatch: got {got_parse}, want {want_parse}"
        elif not diff_ok:
            out["_note"] = f"diff mismatch: got {got_diff}, want {want_diff}"

    except Exception as e:
        out["_note"] = f"raised {type(e).__name__}: {e}"

    return out
