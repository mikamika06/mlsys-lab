import ast


class GraphBreakVisitor(ast.NodeVisitor):
  """AST visitor to count graph breaks in Python model code."""

  def __init__(self):
    raise NotImplementedError


def count_graph_breaks(source_or_fn):
  raise NotImplementedError


def rank_snippets(snippets):
  raise NotImplementedError
