import torch
import ref

def check(workdir):
    out = {"fake_registered": 0.0, "fullgraph_compiled": 0.0}
    try:
        import custom_op.wrapper
        q, k, v = ref.get_test_inputs()

        from torch._subclasses.fake_tensor import FakeTensorMode
        try:
            with FakeTensorMode():
                q_f = torch.empty(2, 4, 16, 32)
                k_f = torch.empty(2, 4, 16, 32)
                v_f = torch.empty(2, 4, 16, 32)
                res_fake = custom_op.wrapper.run_attention(q_f, k_f, v_f)
                if isinstance(res_fake, torch.Tensor) and res_fake.shape == q.shape:
                    out["fake_registered"] = 1.0
        except Exception as fe:
            out["_note"] = f"FakeTensor check failed: {type(fe).__name__}: {str(fe)[:120]}"

        try:
            compiled_fn = torch.compile(custom_op.wrapper.run_attention, fullgraph=True)
            got = compiled_fn(q, k, v)
            want = ref.expected_attention(q, k, v)
            if isinstance(got, torch.Tensor) and torch.allclose(got, want, atol=1e-5):
                out["fullgraph_compiled"] = 1.0
        except Exception as ce:
            out["_note"] = f"Compile check failed: {type(ce).__name__}: {str(ce)[:120]}"

    except Exception as e:
        out["_note"] = f"Failed: {type(e).__name__}: {str(e)[:120]}"
    return out
