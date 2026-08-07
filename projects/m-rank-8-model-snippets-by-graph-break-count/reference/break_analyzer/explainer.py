import ast
import inspect
from break_analyzer.tracer import GraphBreakVisitor, count_graph_breaks


def explain(source_or_fn):
  if callable(source_or_fn):
    source = inspect.getsource(source_or_fn)
  else:
    source = str(source_or_fn)
  tree = ast.parse(source)
  visitor = GraphBreakVisitor()
  visitor.visit(tree)
  return {
      "graph_break_count": visitor.break_count,
      "graph_count": visitor.break_count + 1,
      "break_reasons": visitor.reasons,
      "break_lines": visitor.lines,
  }


def predict_graph_count(source_or_fn):
  return count_graph_breaks(source_or_fn) + 1
