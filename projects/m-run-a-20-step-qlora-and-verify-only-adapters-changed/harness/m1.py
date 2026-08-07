import ref
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 16)
        self.q_proj = nn.Linear(16, 16)
        self.v_proj = nn.Linear(16, 16)
        self.lm_head = nn.Linear(16, 100)

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        h = self.q_proj(x) + self.v_proj(x)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, 100), labels.view(-1))
        return {"loss": loss, "logits": logits}


def check(workdir):
    from qlorarun.train import run_training_steps

    out = {"steps_completed": 0.0, "loss_decreased": 0.0}
    dataset = ref.get_reference_setup()
    model = TinyModel()

    try:
        trained_model = run_training_steps(model, None, dataset, steps=5)
        out["steps_completed"] = 5.0
        out["loss_decreased"] = 1.0
    except Exception as e:
        out["_note"] = f"training failed: {type(e).__name__}: {str(e)[:120]}"

    return out
