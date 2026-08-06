import torch
import ref

def check(workdir):
    out = {"op_registered": 0.0, "eager_correct": 0.0}
    try:
        import custom_op.wrapper
        q, k, v = ref.get_test_inputs()
        want = ref.expected_attention(q, k, v)
        got = custom_op.wrapper.run_attention(q, k, v)
        if isinstance(got, torch.Tensor) and torch.allclose(got, want, atol=1e-5):
            out["eager_correct"] = 1.0
        if hasattr(torch.ops, "custom_flash") and hasattr(torch.ops.custom_flash, "flash_attn"):
            out["op_registered"] = 1.0
    except Exception as e:
        out["_note"] = f"Failed: {type(e).__name__}: {str(e)[:120]}"
    return out
