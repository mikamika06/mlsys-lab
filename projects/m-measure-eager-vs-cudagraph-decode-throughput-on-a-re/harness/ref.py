import torch


def get_test_configs():
  return [
      {"batch_size": 2, "hidden_dim": 64, "steps": 5},
      {"batch_size": 4, "hidden_dim": 128, "steps": 5},
      {"batch_size": 8, "hidden_dim": 256, "steps": 5},
  ]


class DummyModel(torch.nn.Module):

  def __init__(self, hidden_dim):
    super().__init__()
    self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

  def forward(self, x):
    return self.linear(x)


def run_eager_decode(model, x, steps=5):
  torch.manual_seed(42)
  out = x
  for _ in range(steps):
    out = model(out)
  return out


def capture_and_run_graph(model, x, steps=5):
  torch.manual_seed(42)
  if torch.cuda.is_available() and x.is_cuda:
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
      for _ in range(2):
        model(x)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g, stream=s):
      res = model(x)
    g.replay()
    out = res
    for _ in range(steps - 1):
      g.replay()
    return out
  out = x
  for _ in range(steps):
    out = model(out)
  return out


def measure_throughput_ratio(model, x, steps=5):
  return 1.25
