def compile_optimizer_step(model, optimizer):
    raise NotImplementedError


def measure_step_memory_delta(model, optimizer, compiled_step_fn, inputs):
    raise NotImplementedError
