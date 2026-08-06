import ref
import torch


def check(workdir):
  from cudaperf.eager import run_eager_decode

  configs = ref.get_test_configs()
  out = {"eager_matched": 0.0, "configs": float(len(configs))}
  ok = 0
  for i, cfg in enumerate(configs):
    torch.manual_seed(123)
    model = ref.DummyModel(cfg["hidden_dim"])
    x = torch.randn(cfg["batch_size"], cfg["hidden_dim"])
    want = ref.run_eager_decode(model, x, steps=cfg["steps"])
    try:
      got = run_eager_decode(model, x, steps=cfg["steps"])
      if torch.allclose(got, want, atol=1e-5, rtol=1e-5):
        ok += 1
      elif "_note" not in out:
        out["_note"] = (
            f"config {i}: eager output mismatch, got shape {got.shape},"
            f" expected {want.shape}"
        )
    except Exception as e:
      if "_note" not in out:
        out["_note"] = f"config {i} raised {type(e).__name__}: {str(e)[:100]}"
  out["eager_matched"] = float(ok)
  return out
