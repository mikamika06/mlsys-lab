import warnings
import ref

def execute_oneshot(model_stub):
    res = ref.run_oneshot_quantization(model_stub)
    return res
