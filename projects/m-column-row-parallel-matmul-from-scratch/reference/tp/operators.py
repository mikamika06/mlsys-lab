import torch
import torch.distributed as dist


class ColumnParallelMatmulFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, weight, process_group=None):
        ctx.save_for_backward(input_, weight)
        ctx.process_group = process_group
        output = torch.matmul(input_, weight.t())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        process_group = ctx.process_group

        grad_input = torch.matmul(grad_output, weight)
        if dist.is_initialized():
            dist.all_reduce(grad_input, group=process_group)

        grad_weight = torch.matmul(grad_output.transpose(-2, -1), input_)
        return grad_input, grad_weight, None


class RowParallelMatmulFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, weight, process_group=None):
        ctx.save_for_backward(input_, weight)
        ctx.process_group = process_group
        output = torch.matmul(input_, weight.t())
        if dist.is_initialized():
            dist.all_reduce(output, group=process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        grad_input = torch.matmul(grad_output, weight)
        grad_weight = torch.matmul(grad_output.transpose(-2, -1), input_)
        return grad_input, grad_weight, None


def column_parallel_matmul(input_, weight, process_group=None):
    return ColumnParallelMatmulFunction.apply(input_, weight, process_group)


def row_parallel_matmul(input_, weight, process_group=None):
    return RowParallelMatmulFunction.apply(input_, weight, process_group)
