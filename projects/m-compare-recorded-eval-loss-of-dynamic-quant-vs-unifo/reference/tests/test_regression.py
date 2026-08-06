from workflow.eval_compare import compare_eval_loss

def test_eval_loss_comparison():
  dyn = [{"step": 10, "eval_loss": 2.0}]
  uni = [{"step": 10, "eval_loss": 2.5}]
  diff = compare_eval_loss(dyn, uni)
  assert abs(diff - 0.2) < 1e-5, f"Expected 0.2 relative difference, got {diff}"
