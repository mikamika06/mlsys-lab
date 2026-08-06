import tempfile
import os
import time
import numpy as np
import onnx
import onnxruntime as ort


def _get_opts(intra_op_num_threads, opt_level):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_op_num_threads
    if opt_level == "BASIC":
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    elif opt_level == "ALL":
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    else:
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_DEFAULT
    opts.optimized_model_filepath = ""
    return opts


def run_inference(model_bytes, inputs, intra_op_num_threads=1, opt_level="BASIC"):
    opts = _get_opts(intra_op_num_threads, opt_level)
    session = ort.InferenceSession(model_bytes, opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    res = session.run([output_name], {input_name: inputs})
    return res[0]


def get_optimized_node_count(model_bytes, opt_level="BASIC"):
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        opts = _get_opts(1, opt_level)
        opts.optimized_model_filepath = tmp_name
        ort.InferenceSession(model_bytes, opts, providers=["CPUExecutionProvider"])
        opt_model = onnx.load(tmp_name)
        return len(opt_model.graph.node)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def measure_latency_scaling(model_bytes, inputs, thread_counts):
    latencies = {}
    for t in thread_counts:
        run_inference(model_bytes, inputs, intra_op_num_threads=t, opt_level="BASIC")
        start = time.perf_counter()
        for _ in range(30):
            run_inference(model_bytes, inputs, intra_op_num_threads=t, opt_level="BASIC")
        end = time.perf_counter()
        latencies[t] = (end - start) / 30.0
    return latencies
