import numpy as np
import ref


def check(workdir):
    from edgeio.pipeline import compare_pipeline_memory, preprocess_app_side, preprocess_in_graph_node

    out = {"dual_path_correct": 0.0, "overhead_quantified": 0.0}

    raw = ref.sample_frames(batch_size=4, h=32, w=32, c=3, seed=999)

    try:
        app_res = preprocess_app_side(raw, ref.MEAN, ref.STD)
        graph_res = preprocess_in_graph_node(raw, ref.MEAN, ref.STD)

        ref_app = ref.preprocess_app_side(raw, ref.MEAN, ref.STD)
        ref_graph = ref.preprocess_in_graph_node(raw, ref.MEAN, ref.STD)

        app_ok = np.allclose(app_res, ref_app, rtol=1e-5, atol=1e-5)
        graph_ok = np.allclose(graph_res, ref_graph, rtol=1e-5, atol=1e-5)

        if app_ok and graph_ok:
            out["dual_path_correct"] = 1.0
        else:
            out["_note"] = f"App match: {app_ok}, Graph match: {graph_ok}"

        mem_comp = compare_pipeline_memory(4, 32, 32, 3)
        if mem_comp.get("bandwidth_reduction_factor") == 4.0:
            out["overhead_quantified"] = 1.0
        else:
            out["_note"] = f"Unexpected bandwidth reduction factor: {mem_comp.get('bandwidth_reduction_factor')}"

    except Exception as e:
        out["_note"] = f"Error during pipeline evaluation: {type(e).__name__}: {str(e)}"

    return out
