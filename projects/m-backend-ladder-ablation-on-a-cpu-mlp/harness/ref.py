import torch
import torch._dynamo as dynamo

def run_ladder(model, args):
    res = {}
    dynamo.reset()
    res['eager'] = model(*args)
    
    dynamo.reset()
    c_aot = torch.compile(model, backend="aot_eager")
    res['aot_eager'] = c_aot(*args)
    
    dynamo.reset()
    c_ind = torch.compile(model, backend="inductor")
    res['inductor'] = c_ind(*args)
    
    return res

def explain_dynamo(model, args):
    dynamo.reset()
    explanation = dynamo.explain(model)(*args)
    return explanation.graph_count, explanation.graph_break_count

def parse_graph_code(code_str):
    ops = []
    for line in code_str.split('\n'):
        line = line.strip()
        if '=' in line and '(' in line:
            rhs = line.split('=', 1)[1].strip()
            op = rhs.split('(')[0].strip()
            if op.startswith("torch.ops."):
                ops.append(op)
    return ops
