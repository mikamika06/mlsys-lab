import ast
import inspect


class GraphBreakVisitor(ast.NodeVisitor):
  """AST visitor to count graph breaks in Python model code."""

  def __init__(self):
    self.break_count = 0
    self.reasons = []
    self.lines = []

  def _is_constant(self, node):
    if isinstance(node, ast.Constant):
      return True
    return False

  def visit_If(self, node):
    if not self._is_constant(node.test):
      self.break_count += 1
      self.reasons.append("dynamic_if")
      self.lines.append(getattr(node, "lineno", 0))
    self.generic_visit(node)

  def visit_While(self, node):
    if not self._is_constant(node.test):
      self.break_count += 1
      self.reasons.append("dynamic_while")
      self.lines.append(getattr(node, "lineno", 0))
    self.generic_visit(node)

  def visit_Try(self, node):
    self.break_count += 1
    self.reasons.append("unsupported_try_except")
    self.lines.append(getattr(node, "lineno", 0))
    self.generic_visit(node)

  def visit_Call(self, node):
    name = ""
    if isinstance(node.func, ast.Name):
      name = node.func.id
    elif isinstance(node.func, ast.Attribute):
      name = node.func.attr

    if name in ("print", "graph_break", "exit"):
      self.break_count += 1
      self.reasons.append(f"unsupported_call_{name}")
      self.lines.append(getattr(node, "lineno", 0))
    elif isinstance(node.func, ast.Attribute) and isinstance(
        node.func.value, ast.Name
    ):
      if node.func.value.id in ("logging", "sys"):
        self.break_count += 1
        self.reasons.append(f"unsupported_call_{node.func.value.id}")
        self.lines.append(getattr(node, "lineno", 0))

    self.generic_visit(node)


def count_graph_breaks(source_or_fn):
  if callable(source_or_fn):
    source = inspect.getsource(source_or_fn)
  else:
    source = str(source_or_fn)
  tree = ast.parse(source)
  visitor = GraphBreakVisitor()
  visitor.visit(tree)
  return visitor.break_count


def rank_snippets(snippets):
  items = []
  for i, s in enumerate(snippets):
    cnt = count_graph_breaks(s)
    items.append({"index": i, "break_count": cnt, "code": s})
  items.sort(key=lambda x: (-x["break_count"], x["index"]))
  return items
