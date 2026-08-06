import ref

def check(workdir):
  from workflow.eval_compare import compare_eval_loss

  out = {"eval_loss_rel_err": 1.0}
  want = ref.compare_eval_loss(ref.LOG_DATA["dynamic"], ref.LOG_DATA["uniform"])
  got = compare_eval_loss(ref.LOG_DATA["dynamic"], ref.LOG_DATA["uniform"])
  rel_err = abs(got - want) / abs(want) if want != 0 else abs(got)
  out["eval_loss_rel_err"] = float(rel_err)
  if rel_err > 1e-4:
    out["_note"] = f"Expected mean relative difference {want}, got {got}"
  return out
