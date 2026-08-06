import sys
import torch
import ref

class BreakModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(10, 10)
        
    def forward(self, x):
        y = self.lin(x)
        z = y.tolist()
        return torch.tensor(z)

def check(workdir):
    sys.path.insert(0, workdir)
    out = {"ladder_exact_match": 0.0, "explain_exact_match": 0.0}
    
    try:
        from ablation.ladder import run_ladder, explain_dynamo
        
        torch.manual_seed(42)
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 5)
        )
        x = torch.randn(2, 10)
        
        try:
            got_ladder = run_ladder(model, (x,))
            want_ladder = ref.run_ladder(model, (x,))
            if list(got_ladder.keys()) == ['eager', 'aot_eager', 'inductor']:
                match = True
                for k in want_ladder:
                    if not torch.allclose(got_ladder[k], want_ladder[k], atol=1e-4):
                        match = False
                if match:
                    out["ladder_exact_match"] = 1.0
        except Exception as e:
            out["_note_ladder"] = str(e)
            
        bm = BreakModel()
        try:
            got_explain = explain_dynamo(bm, (x,))
            want_explain = ref.explain_dynamo(bm, (x,))
            if got_explain == want_explain:
                out["explain_exact_match"] = 1.0
            else:
                out["_note_explain"] = f"got {got_explain}, want {want_explain}"
        except Exception as e:
            out["_note_explain"] = str(e)
            
    finally:
        sys.path.pop(0)
        
    return out
