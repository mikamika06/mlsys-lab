import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from jaxinspect.ir import analyze_stablehlo_ir

    out = {"ir_parsed_correctly": 0.0}

    res = analyze_stablehlo_ir(ref.STABLEHLO_IR_SAMPLE)
    ref_res = ref.analyze_stablehlo_ir(ref.STABLEHLO_IR_SAMPLE)

    if res == ref_res:
        out["ir_parsed_correctly"] = 1.0
    else:
        out["_note"] = f"IR parsing mismatched: got {res}, expected {ref_res}"

    return out
