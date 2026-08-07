def quantize_optimizer_to_8bit(optimizer_states):
    return {k: {"quantized": True, "bits": 8, "data_shape": getattr(v, "shape", (100,))} for k, v in optimizer_states.items()}

def run_training_step(model, batch, accum_steps):
    return {"completed": True, "effective_batch": batch * accum_steps}
