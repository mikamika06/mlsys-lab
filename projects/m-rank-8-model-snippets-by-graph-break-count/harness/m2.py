import sys
import ref


def check(workdir):
  if workdir not in sys.path:
    sys.path.insert(0, workdir)

  from break_analyzer.explainer import explain, predict_graph_count

  out = {"predictions_matched": 0.0, "explain_accuracy": 0.0}

  funcs = ref.NESTED_IF_FUNCTIONS

  pred_ok = True
  explain_ok = True

  for i, fn_src in enumerate(funcs):
    ref_exp = ref.explain(fn_src)
    ref_pred = ref.predict_graph_count(fn_src)

    try:
      got_pred = predict_graph_count(fn_src)
      if got_pred != ref_pred:
        pred_ok = False
        out["_note"] = (
            f"func {i}: predicted graph_count {got_pred}, expected {ref_pred}"
        )
    except Exception as e:
      pred_ok = False
      out["_note"] = f"predict_graph_count raised {e}"

    try:
      got_exp = explain(fn_src)
      if (
          got_exp.get("graph_break_count") != ref_exp["graph_break_count"]
          or got_exp.get("graph_count") != ref_exp["graph_count"]
      ):
        explain_ok = False
        out["_note"] = f"func {i}: explain returned {got_exp}, want {ref_exp}"
    except Exception as e:
      explain_ok = False
      out["_note"] = f"explain raised {e}"

  out["predictions_matched"] = 1.0 if pred_ok else 0.0
  out["explain_accuracy"] = 1.0 if explain_ok else 0.0

  return out
