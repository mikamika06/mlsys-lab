class MultiAdapterLinear:
    def __init__(self, in_features, out_features, base_weight=None):
        raise NotImplementedError

    def add_adapter(self, adapter_name, r, lora_alpha, lora_A=None, lora_B=None):
        raise NotImplementedError

    def set_adapter(self, adapter_name):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
