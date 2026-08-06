import torch


def find_experimental_backends() -> list[str]:
    """Find backends present when exclude_tags=[] but omitted by default."""
    default_backends = set(torch._dynamo.list_backends())
    all_backends = set(torch._dynamo.list_backends(exclude_tags=[]))
    experimental = sorted(list(all_backends - default_backends))
    return experimental
