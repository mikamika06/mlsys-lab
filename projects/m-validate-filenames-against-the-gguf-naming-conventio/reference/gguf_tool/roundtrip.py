import torch

def gguf_to_torch(gguf_data: dict) -> dict:
    tensors = gguf_data.get("tensors", {})
    return {k: torch.tensor(v) for k, v in tensors.items()}


def torch_to_gguf(state_dict: dict, metadata: dict) -> dict:
    tensors = {k: v.detach().cpu().numpy().tolist() for k, v in state_dict.items()}
    return {"metadata": metadata, "tensors": tensors}
