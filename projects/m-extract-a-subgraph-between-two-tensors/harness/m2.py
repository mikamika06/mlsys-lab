import ref
import copy


def check(workdir):
    from optimizer.fuse import fuse_gelu
    from optimizer.constant import fold_initializer_constants

    model = ref.build_fuse_model()
    unused_init = ref.onnx.helper.make_tensor("unused_init", ref.onnx.TensorProto.FLOAT, [1], [1.0])
    model.graph.initializer.append(unused_init)

    fused = fuse_gelu(copy.deepcopy(model))
    folded = fold_initializer_constants(copy.deepcopy(model))

    out = {"fuse_matched": 0.0, "constant_fold_matched": 0.0}

    has_gelu = any(n.op_type == "Gelu" for n in fused.graph.node)
    if has_gelu:
        out["fuse_matched"] = 1.0
    else:
        out["_note"] = "Gelu fusion pattern not detected in graph nodes."

    has_unused = any(init.name == "unused_init" for init in folded.graph.initializer)
    if not has_unused:
        out["constant_fold_matched"] = 1.0
    else:
        out["_note"] = "Unused initializers were not folded/removed."

    return out
