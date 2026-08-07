class Core:
    """Core OpenVINO runtime context."""

    def compile_model(self, model_config):
        raise NotImplementedError


class CompiledModel:
    """Compiled model ready for synchronous or asynchronous inference requests."""

    def __init__(self, model_config):
        raise NotImplementedError

    def create_infer_request(self):
        raise NotImplementedError

    def __call__(self, inputs):
        raise NotImplementedError
