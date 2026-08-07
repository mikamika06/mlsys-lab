import torch


class CUDAGraphRunner:
    def __init__(self, model):
        self.model = model
        self.graph = None
        self.static_input = None
        self.static_output = None

    def capture(self, x):
        self.static_input = x.clone()
        self.static_output = torch.empty_like(x)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = self.model(self.static_input)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=s):
            self.static_output.copy_(self.model(self.static_input))

    def run(self, x):
        if self.graph is None:
            return self.model(x)
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output.clone()
