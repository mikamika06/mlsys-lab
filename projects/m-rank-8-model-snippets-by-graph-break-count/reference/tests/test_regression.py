import sys

sys.path.insert(0, ".")
from break_analyzer.explainer import explain, predict_graph_count
from break_analyzer.tracer import count_graph_breaks, rank_snippets

SNIPPET = """
def sample_func(x):
    print("tracing sample")
    if x.sum() > 0:
        x = x + 1
    return x
"""


def test_count_graph_breaks_detects_breaks():
  cnt = count_graph_breaks(SNIPPET)
  assert cnt == 2, f"Expected 2 graph breaks, got {cnt}"


def test_explain_matches_prediction():
  exp = explain(SNIPPET)
  pred = predict_graph_count(SNIPPET)
  assert exp["graph_count"] == pred
  assert exp["graph_break_count"] == 2


def test_ranking_order_descending():
  snippets = [
      "def f0(x):\n    return x",
      "def f1(x):\n    print('hi')\n    if x > 0:\n        return x",
  ]
  ranked = rank_snippets(snippets)
  assert len(ranked) == 2
  assert ranked[0]["break_count"] >= ranked[1]["break_count"]
