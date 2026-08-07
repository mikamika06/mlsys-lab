from autoround.quant import learned_round_layer


class AutoRoundModifier:
    def __init__(self, steps=100, lr=0.1, bits=4):
        self.steps = steps
        self.lr = lr
        self.bits = bits

    def fit_transform(self, layers, calibration_data):
        optimized_layers = []
        cur_X = calibration_data
        for layer in layers:
            W = layer["weight"]
            W_q, scale, zp = learned_round_layer(
                W, cur_X, steps=self.steps, lr=self.lr, bits=self.bits
            )
            optimized_layers.append({"weight": W_q, "scale": scale, "zp": zp})
            cur_X = cur_X @ W_q.T
        return optimized_layers
