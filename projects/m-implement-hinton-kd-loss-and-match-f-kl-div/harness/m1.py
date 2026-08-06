import torch
import ref


def check(workdir):
    from distill.loss import hinton_kd_loss

    cases = ref.get_test_cases()
    matched = 0.0
    for i, (s, t, temp, alpha, labels) in enumerate(cases):
        want = ref.ref_hinton_kd_loss(s, t, temperature=temp, alpha=alpha, labels=labels)
        try:
            got = hinton_kd_loss(s, t, temperature=temp, alpha=alpha, labels=labels)
        except Exception as e:
            return {"loss_matched": 0.0, "_note": f"case {i} raised {type(e).__name__}: {e}"}

        if torch.allclose(got, want, atol=1e-4, rtol=1e-4):
            matched += 1.0
        elif matched == 0:
            return {"loss_matched": 0.0, "_note": f"case {i} mismatch: got {got.item()}, want {want.item()}"}

    return {"loss_matched": matched}
