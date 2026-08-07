class Engine:
    def __init__(self, shards):
        self.shards = shards
        self.index = {}
        for i, s in enumerate(shards):
            for k in s.tensors.keys():
                self.index[k] = i

    def forward(self, x, layer_names):
        y = x.copy()
        for name in layer_names:
            shard_idx = self.index[name]
            w = self.shards[shard_idx].tensors[name]
            y = y @ w
        return y
