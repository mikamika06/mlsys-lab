import os
import ref

def check(workdir):
    m = {"accelerator_share": 0.0}
    path = ref.create_dummy_model(workdir)
    try:
        from edge.model import analyze_graph
        st = analyze_graph(path)
        total = st.get("total_ops", 1)
        delegated = st.get("delegated_ops", 0)
        if total > 0:
            m["accelerator_share"] = float(delegated) / float(total)
    except Exception:
        pass
    return m
