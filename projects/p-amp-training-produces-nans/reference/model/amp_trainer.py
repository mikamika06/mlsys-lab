import torch


def locate_first_nan(model, inputs):
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            return name
    return None


def check_autocast_ops(model, inputs):
    dtypes = {}
    with torch.autocast(device_type="cpu"):
        out = model(inputs)
        if isinstance(out, torch.Tensor):
            dtypes["output"] = out.dtype
    return dtypes


def simulate_grad_scaler(scaler, loss):
    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()
    scaler.step(torch.optim.SGD(torch.nn.Parameter(torch.tensor([1.0])), lr=0.1))
    scaler.update()
    return scaler.get_scale()


def isolate_sensitive_layers(model):
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.float()
    return model


def train_stable_steps(model, dataloader, steps=1000):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    for _ in range(steps):
        for x, y in dataloader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=False):
                out = model(x)
                loss = torch.nn.functional.mse_loss(out, y)
            loss.backward()
            optimizer.step()
    return True


class NaNDetector:
    def __init__(self, model):
        self.model = model
        self.hooks = []

    def register(self):
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor) and torch.isnan(output).any():
                    raise ValueError(f"NaN detected in module {name}")
            return hook
        for name, module in self.model.named_modules():
            self.hooks.append(module.register_forward_hook(make_hook(name)))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
