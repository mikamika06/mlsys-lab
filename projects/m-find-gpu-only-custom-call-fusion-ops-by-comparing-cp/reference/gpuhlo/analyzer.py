import re


def parse_hlo_module(hlo_text):
    computations = {}
    current_comp = None
    current_lines = []
    for line in hlo_text.splitlines():
        comp_match = re.match(r'HloModule\s+(\w+)', line)
        if comp_match:
            continue
        begin_match = re.match(r'(\w+)\s*\{', line)
        if begin_match and not current_comp:
            current_comp = begin_match.group(1)
            current_lines = [line]
            continue
        if current_comp:
            current_lines.append(line)
            if line.strip() == "}":
                computations[current_comp] = "\n".join(current_lines)
                current_comp = None
                current_lines = []
    return {"computations": computations, "raw": hlo_text}


def extract_computations_and_ops(module_ast):
    ops = []
    comp_dict = module_ast.get("computations", {})
    for comp_name, text in comp_dict.items():
        for line in text.splitlines():
            line_str = line.strip()
            if "=" in line_str and not line_str.startswith("//"):
                parts = line_str.split("=", 1)
                lhs = parts[0].strip()
                rhs = parts[1].strip()
                op_match = re.match(r'(\w+)\s*\(', rhs)
                if op_match:
                    op_name = op_match.group(1)
                    ops.append({"computation": comp_name, "op": op_name, "expression": rhs})
    return ops
