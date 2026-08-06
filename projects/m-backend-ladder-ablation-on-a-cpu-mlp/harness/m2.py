import sys
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    out = {"parser_exact_match": 0.0}
    
    sample_code = """
def forward(self, arg0_1, arg1_1):
    add = torch.ops.aten.add.Tensor(arg0_1, arg1_1)
    relu = torch.ops.aten.relu.default(add)
    view = torch.ops.aten.view.default(relu, [2, 5])
    return (view,)
    """
    
    try:
        from ablation.parser import parse_graph_code
        got = parse_graph_code(sample_code)
        want = ref.parse_graph_code(sample_code)
        
        if got == want:
            out["parser_exact_match"] = 1.0
        else:
            out["_note"] = f"got {got}, want {want}"
            
    except Exception as e:
        out["_note"] = str(e)
    finally:
        sys.path.pop(0)
        
    return out
