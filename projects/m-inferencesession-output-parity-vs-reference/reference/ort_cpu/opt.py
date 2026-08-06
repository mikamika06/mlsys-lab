import onnx
import onnxruntime as ort


def compare_optimization_levels(model_bytes):
    results = {}
    for name, level in [("basic", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
                        ("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL)]:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = level
        session = ort.InferenceSession(model_bytes, opts, providers=['CPUExecutionProvider'])
        model_proto = onnx.load_from_string(session.get_model_bytes())
        results[name] = len(model_proto.graph.node)
    return results
