import torch


def capture_sparse_matmul_error():
    try:
        a = torch.randn(16, 16)
        b = torch.randn(16, 16)
        sparse_a = torch.sparse.to_sparse_semi_structured(a)
        torch.mm(sparse_a, b)
    except Exception as e:
        return type(e), str(e)
    return None, ""
