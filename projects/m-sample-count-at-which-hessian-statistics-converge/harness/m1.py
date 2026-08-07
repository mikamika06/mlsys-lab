import ref


def check(workdir):
  from quant.loader import DeterministicLoader

  data = ref.generate_data(42, 100, 8)
  out = {"loader_deterministic": 0.0}
  try:
    l1 = DeterministicLoader(data, seed=123, batch_size=16)
    l2 = DeterministicLoader(data, seed=123, batch_size=16)
    b1 = list(l1)
    b2 = list(l2)
    if len(b1) == len(b2) and all((x == y).all() for x, y in zip(b1, b2)):
      out["loader_deterministic"] = 1.0
  except Exception as e:
    out["_note"] = f"Error: {e}"
  return out
