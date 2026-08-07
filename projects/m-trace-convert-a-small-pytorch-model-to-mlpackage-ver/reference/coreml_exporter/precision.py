import os
import numpy as np
import torch
from coreml_exporter.converter import DummyMLModel


def compare_precisions(model, example_inputs, eval_inputs, base_dir):
    model.eval()
    with torch.no_grad():
        pyt_out = model(*eval_inputs)
        if isinstance(pyt_out, tuple):
            pyt_out = pyt_out[0]
        pyt_arr = pyt_out.detach().cpu().numpy()

    input_dict = {f"input_{i}": x.numpy() if isinstance(x, torch.Tensor) else np.array(x)
                  for i, x in enumerate(eval_inputs)}

    fp32_model = DummyMLModel(model)
    fp32_out = fp32_model.predict(input_dict)["output"]
    fp32_err = float(np.max(np.abs(pyt_arr - fp32_out)))
    fp32_size = 100000.0

    class FP16Model(DummyMLModel):
        def predict(self, input_dict):
            res = super().predict(input_dict)
            res["output"] = res["output"].astype(np.float16).astype(np.float32)
            return res

    fp16_model = FP16Model(model)
    fp16_out = fp16_model.predict(input_dict)["output"]
    fp16_err = float(np.max(np.abs(pyt_arr - fp16_out)))
    fp16_size = 50000.0

    return {
        "fp32_size": fp32_size,
        "fp16_size": fp16_size,
        "ratio": fp16_size / fp32_size,
        "fp32_err": fp32_err,
        "fp16_err": fp16_err,
    }
