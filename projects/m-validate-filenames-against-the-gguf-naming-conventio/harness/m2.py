import ref
import torch

def check(workdir):
    from gguf_tool.roundtrip import gguf_to_torch, torch_to_gguf
    out = {"roundtrip_matched": 0.0}
    try:
        tensors_in = {k: torch.tensor(v) for k, v in ref.SAMPLE_GGUF["tensors"].items()}
        gguf_built = torch_to_gguf(tensors_in, ref.SAMPLE_GGUF["metadata"])
        recovered = gguf_to_torch(gguf_built)

        if "token_embd.weight" in recovered:
            orig = tensors_in["token_embd.weight"]
            rec = recovered["token_embd.weight"]
            if torch.equal(orig, rec):
                out["roundtrip_matched"] = 1.0
            else:
                out["_note"] = "Round-trip tensor values do not match original tensors"
        else:
            out["_note"] = "Round-trip output missing expected tensor keys"
    except Exception as e:
        out["_note"] = f"Round-trip raised exception: {str(e)[:100]}"
    return out
