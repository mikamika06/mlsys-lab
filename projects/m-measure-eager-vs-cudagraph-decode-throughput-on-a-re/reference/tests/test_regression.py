import sys
import torch

sys.path.insert(0, ".")
from cudaperf.eager import run_eager_decode
from cudaperf.graph import capture_and_run_graph
from cudaperf.measure import measure_throughput_ratio


def test_throughput_ratio_positive():
  model = torch.nn.Linear(32, 32)
  x = torch.randn(2, 32)
  ratio = measure_throughput_ratio(model, x, steps=3)
  assert ratio > 0.0


def test_eager_output_shape():
  model = torch.nn.Linear(32, 32)
  x = torch.randn(2, 32)
  out = run_eager_decode(model, x, steps=2)
  assert out.shape == (2, 32)


def test_graph_output_shape():
  model = torch.nn.Linear(32, 32)
  x = torch.randn(2, 32)
  out = capture_and_run_graph(model, x, steps=2)
  assert out.shape == (2, 32)
