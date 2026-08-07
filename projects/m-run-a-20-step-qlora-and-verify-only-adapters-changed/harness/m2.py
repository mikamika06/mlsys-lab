import ref
import torch
import torch.nn as nn


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
    from qlorarun.verify import verify_only_adapters_changed

    out = {"base_weights_unchanged": 0.0, "adapters_changed": 0.0}

    model = TinyModel()
    initial_state = {name: param.clone() for name, param in model.named_parameters()}

    # Simulate adapter change and base unchanged
    final_state = {name: param.clone() for name, param in model.named_parameters()}
    for name, param in final_state.items():
        if "q_proj" in name:
            param.data += 0.01  # Mocking adapter update

    res = verify_only_adapters_changed(initial_state, final_state, model)
    if res:
        out["base_weights_unchanged"] = 1.0
        out["adapters_changed"] = 1.0

    return out
