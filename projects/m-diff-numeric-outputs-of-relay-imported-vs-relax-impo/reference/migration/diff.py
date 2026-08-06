import numpy as np
import ref


def compare_relay_relax_outputs(model_spec):
    relay_out = ref.compute_relay_output(model_spec)
    relax_out = ref.compute_relax_output(model_spec)
    return float(np.max(np.abs(relay_out - relax_out)))
