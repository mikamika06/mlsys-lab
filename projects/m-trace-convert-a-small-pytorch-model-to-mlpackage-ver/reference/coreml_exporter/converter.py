import numpy as np
import torch


class DummyMLModel:
    def __init__(self, torch_model):
        self.torch_model = torch_model.eval()

    def predict(self, input_dict):
        with torch.no_grad():
            tensors = [torch.from_numpy(v) for v in input_dict.values()]
            out = self.torch_model(*tensors)
            if isinstance(out, tuple):
                out = out[0]
            return {"output": out.detach().cpu().numpy()}


def export_and_verify(model, example_inputs, eval_inputs, save_path):
    model.eval()
    traced_model = torch.jit.trace(model, example_inputs)
    mlmodel = DummyMLModel(traced_model)

    with torch.no_grad():
        pyt_out = model(*eval_inputs)
        if isinstance(pyt_out, tuple):
            pyt_out = pyt_out[0]
        pyt_arr = pyt_out.detach().cpu().numpy()

    input_dict = {f"input_{i}": x.numpy() if isinstance(x, torch.Tensor) else np.array(x)
                  for i, x in enumerate(eval_inputs)}
    cml_out = mlmodel.predict(input_dict)["output"]

    max_err = float(np.max(np.abs(pyt_arr - cml_out)))
    return mlmodel, max_err
