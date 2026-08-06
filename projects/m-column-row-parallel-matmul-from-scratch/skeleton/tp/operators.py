import torch


class ColumnParallelMatmulFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, weight, process_group=None):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class RowParallelMatmulFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, weight, process_group=None):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


def column_parallel_matmul(input_, weight, process_group=None):
    raise NotImplementedError


def row_parallel_matmul(input_, weight, process_group=None):
    raise NotImplementedError
