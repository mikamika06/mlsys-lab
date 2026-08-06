class ModelConverter:
    def __init__(self, config: dict, state_dict: dict):
        self.config = config
        self.state_dict = state_dict

    def get_metadata(self) -> dict:
        raise NotImplementedError

    def get_tensors(self) -> dict:
        raise NotImplementedError


class MiniGPTConverter(ModelConverter):
    def get_metadata(self) -> dict:
        raise NotImplementedError

    def get_tensors(self) -> dict:
        raise NotImplementedError
