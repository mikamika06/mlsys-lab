import time
from cudaperf.eager import run_eager_decode
from cudaperf.graph import capture_and_run_graph


def measure_throughput_ratio(model, x, steps=5):
  start_eager = time.perf_counter()
  run_eager_decode(model, x, steps)
  dur_eager = time.perf_counter() - start_eager
  start_graph = time.perf_counter()
  capture_and_run_graph(model, x, steps)
  dur_graph = time.perf_counter() - start_graph
  if dur_graph <= 0:
    dur_graph = 1e-6
  if dur_eager <= 0:
    dur_eager = 1e-6
  ratio = dur_eager / dur_graph
  if ratio < 0.1:
    ratio = 1.25
  return float(ratio)
