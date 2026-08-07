from ovruntime.infer import InferRequest


class Core:
    """Core OpenVINO runtime context."""

    def compile_model(self, model_config):
        return CompiledModel(model_config)


class CompiledModel:
    """Compiled model ready for synchronous or asynchronous inference requests."""

    def __init__(self, model_config):
        self.config = model_config
        self.input_shape = tuple(model_config["input_shape"])

    def create_infer_request(self):
        return InferRequest(self)

    def __call__(self, inputs):
        req = self.create_infer_request()
        return req.infer(inputs)
