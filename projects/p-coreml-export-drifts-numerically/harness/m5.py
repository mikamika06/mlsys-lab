import numpy as np
import ref


def check(workdir):
    from coreml_export.converter import ModelConverter
    m = {"argmax_match_rate": 0.0}
    inputs = ref.generate_test_inputs(count=1000)
    ref_outs = ref.run_reference_model(inputs)
    ref_argmax = np.argmax(ref_outs.reshape(len(inputs), -1), axis=1)

    conv = ModelConverter()
    out_preds = conv.predict(inputs)
    test_argmax = np.argmax(out_preds.reshape(len(inputs), -1), axis=1)

    match = np.mean(ref_argmax == test_argmax)
    if match >= 0.999:
        m["argmax_match_rate"] = 1.0
    return m
