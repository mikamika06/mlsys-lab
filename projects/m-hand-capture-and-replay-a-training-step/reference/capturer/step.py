import torch


class CapturedStep:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.graph = None
        self.static_inputs = None
        self.static_targets = None
        self.static_loss = None
        self.static_outputs = None

    def capture(self, sample_inputs, sample_targets):
        self.static_inputs = sample_inputs.detach().clone().requires_grad_(sample_inputs.requires_grad)
        self.static_targets = sample_targets.detach().clone()

        self.optimizer.zero_grad(set_to_none=True)

        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                outputs = self.model(self.static_inputs)
                loss = self.loss_fn(outputs, self.static_targets)
                loss.backward()
        torch.cuda.current_stream().wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()

        # Capture
        self.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(self.graph):
            outputs = self.model(self.static_inputs)
            loss = self.loss_fn(outputs, self.static_targets)
            loss.backward()
            self.optimizer.step()
            self.static_outputs = outputs
            self.static_loss = loss

    def replay(self, inputs, targets):
        self.static_inputs.copy_(inputs)
        self.static_targets.copy_(targets)
        self.graph.replay()
        return self.static_outputs.detach(), self.static_loss.detach()
