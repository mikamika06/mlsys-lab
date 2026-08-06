import torch


def capture_and_run_graph(model, x, steps=5):
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
