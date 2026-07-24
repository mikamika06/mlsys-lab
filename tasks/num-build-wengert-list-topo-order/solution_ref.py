def build_wengert_list(output_node):
    seen = set()
    tape = []

    def visit(node):
        marker = id(node)
        if marker in seen:
            return
        seen.add(marker)
        for inp in getattr(node, "inputs", []):
            visit(inp)
        tape.append(
            {
                "name": node.name,
                "op": node.op,
                "inputs": [x.name for x in getattr(node, "inputs", [])],
            }
        )

    visit(output_node)
    return tape
