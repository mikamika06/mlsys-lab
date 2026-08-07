import ast
import inspect

SNIPPETS = [
    """
def snippet_0(x, y):
    a = x + y
    return torch.matmul(a, x)
""",
    """
def snippet_1(x):
    a = x * 2
    if x.sum() > 0:
        a = a + 1
    return a
""",
    """
def snippet_2(x):
    print("Processing tensor")
    a = x + 1
    if x.mean() < 0:
        a = a * 2
    return a
""",
    """
def snippet_3(x, y):
    if x.max() > 0:
        x = x + 1
    try:
        y = y / 2
    except Exception:
        y = y
    if y.min() < 0:
        y = y * 3
    return x + y
""",
    """
def snippet_4(x):
    print("start")
    if x.sum() > 0:
        x = x + 1
    print("mid")
    if x.sum() < 10:
        x = x * 2
    return x
""",
    """
def snippet_5(x, y):
    print("step 1")
    if x.sum() > 0:
        x = x + 1
    if y.sum() > 0:
        y = y + 1
    try:
        x = x * y
    except Exception:
        x = y
    if x.mean() > 0:
        x = x - 1
    return x
""",
    """
def snippet_6(x):
    print("a")
    if x.sum() > 0:
        x = x + 1
    if x.mean() > 1:
        x = x + 2
    print("b")
    if x.min() < 0:
        x = x + 3
    if x.max() > 5:
        x = x + 4
    return x
""",
    """
def snippet_7(x, y):
    print("start")
    if x.sum() > 0:
        x = x + 1
    if x.mean() > 0:
        x = x * 2
    if y.sum() > 0:
        y = y + 1
    print("mid")
    if y.mean() > 0:
        y = y * 2
    if (x + y).sum() > 0:
        x = x + y
    return x
""",
]

NESTED_IF_FUNCTIONS = [
    """
def nested_f1(x, y):
    if x > 0:
        if y > 0:
            return x + y
    return x - y
""",
    """
def nested_f2(a, b, c):
    if a.sum() > 0:
        a = a + 1
        if b.sum() > 0:
            b = b + 1
            if c.sum() > 0:
                c = c + 1
    return a + b + c
""",
    """
def nested_f3(x):
    if x.mean() > 0:
        x = x * 2
    else:
        if x.min() < 0:
            x = x / 2
    return x
""",
    """
def nested_f4(x, y, z):
    if x.max() > 0:
        print("x is positive")
        if y.max() > 0:
            if z.max() > 0:
                print("z is positive")
    return x + y + z
""",
]


class GraphBreakVisitor(ast.NodeVisitor):

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
