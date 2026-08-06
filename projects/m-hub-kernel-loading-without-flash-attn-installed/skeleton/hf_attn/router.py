class AttentionInterface:
    def __init__(self, name, available=True):
        self.name = name
        self.available = available

    def forward(self, q, k, v):
        raise NotImplementedError


def resolve_backend(config, available_backends):
    raise NotImplementedError


def dispatch_attention(config, q, k, v, available_backends):
    raise NotImplementedError
