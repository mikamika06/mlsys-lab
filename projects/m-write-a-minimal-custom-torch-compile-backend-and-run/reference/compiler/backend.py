import torch


def minimal_backend(gm, example_inputs):
    def compiler_fn(g, inputs):
        return g.forward

    return compiler_fn
