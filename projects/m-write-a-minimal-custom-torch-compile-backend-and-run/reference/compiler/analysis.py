import torch


def count_op_frequencies(model, example_inputs):
    class OpCounter(torch.fx.Interpreter):

        def __init__(self, module):
            super().__init__(module)
            self.counts = {}

        def call_function(self, target, args, kwargs):
            name = str(target)
            self.counts[name] = self.counts.get(name, 0) + 1
            return super().call_function(target, args, kwargs)

        def call_method(self, target, args, kwargs):
            name = str(target)
            self.counts[name] = self.counts.get(name, 0) + 1
            return super().call_method(target, args, kwargs)

        def call_module(self, target, args, kwargs):
            name = str(target)
            self.counts[name] = self.counts.get(name, 0) + 1
            return super().call_module(target, args, kwargs)

    fx_graph = torch.fx.symbolic_trace(model)
    interpreter = OpCounter(fx_graph)
    interpreter.run(*example_inputs)
    return interpreter.counts
