def run_eager_decode(model, x, steps=5):
  out = x
  for _ in range(steps):
    out = model(out)
  return out
