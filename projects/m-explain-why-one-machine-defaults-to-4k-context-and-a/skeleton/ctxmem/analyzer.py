def determine_default_context(vram_bytes, model_params):
    raise NotImplementedError


def allocate_slots(num_ctx, num_parallel):
    raise NotImplementedError


def predict_resident_bytes(num_ctx, num_parallel, config):
    raise NotImplementedError
