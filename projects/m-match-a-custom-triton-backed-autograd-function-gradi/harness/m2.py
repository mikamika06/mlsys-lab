import ref
import torch


def check(workdir):
    try:
        from triton_autograd.func import TritonSiluFunction
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"failed to import TritonSiluFunction: {e}"}

    max_err = 0.0
    for x, grad_out in ref.get_test_tensors():
        try:
            x_learner = x.clone().detach().requires_grad_(True)
            y_learner = TritonSiluFunction.apply(x_learner)
            y_learner.backward(grad_out)
            got_grad = x_learner.grad

            want_grad = ref.torch_silu_backward(x, grad_out)

            diff = torch.norm(got_grad - want_grad)
            norm = torch.norm(want_grad)
            err = (diff / (norm + 1e-8)).item()
            if err > max_err:
                max_err = err
        except Exception as e:
            return {"rel_err": 1.0, "_note": f"TritonSiluFunction backward execution failed: {e}"}

    return {"rel_err": float(max_err)}
