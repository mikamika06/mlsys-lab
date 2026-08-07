import numpy as np


class Parameter:

    def __init__(self, data):
        self.data = np.array(data, dtype=np.float64)
        self.grad = None


class DummyOptimizer:

    def __init__(self, param_groups):
        self.param_groups = param_groups


class GradScaler:

    def __init__(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000):
        self._scale = float(init_scale)
        self._growth_factor = float(growth_factor)
        self._backoff_factor = float(backoff_factor)
        self._growth_interval = int(growth_interval)
        self._unscaled_optimizers = set()

    def get_scale(self):
        return self._scale

    def unscale_(self, optimizer):
        opt_id = id(optimizer)
        if opt_id in self._unscaled_optimizers:
            raise RuntimeError("unscale_ has already been called on this optimizer.")

        inv_scale = 1.0 / self._scale
        unscaled_grads = []
        for group in optimizer.param_groups:
            group_grads = []
            for p in group["params"]:
                if p.grad is not None:
                    p.grad = p.grad * inv_scale
                    group_grads.append(p.grad.copy())
                else:
                    group_grads.append(None)
            unscaled_grads.append(group_grads)

        self._unscaled_optimizers.add(opt_id)
        return unscaled_grads
