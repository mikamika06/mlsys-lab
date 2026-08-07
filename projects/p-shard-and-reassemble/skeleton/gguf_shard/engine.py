class Engine:
    def __init__(self, shards):
        raise NotImplementedError

    def forward(self, x, layer_names):
        raise NotImplementedError
