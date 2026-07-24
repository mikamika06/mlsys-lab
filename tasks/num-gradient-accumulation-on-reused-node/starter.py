import numpy as np


def tape_grad(tape: list[tuple[str, tuple[int, ...]]], x: np.ndarray) -> np.ndarray:
    """
    Reverse-mode autograd over a Wengert list.

    ``tape`` is a list of ``(op, input_node_indices)`` entries, one per
    computed node, already in topological (forward-evaluation) order.
    Node indices ``0 .. len(x)-1`` refer to the entries of ``x``; node
    index ``len(x) + i`` refers to the value produced by ``tape[i]``.
    Supported ops: ``"add"``, ``"sub"``, ``"mul"`` (2 inputs each) and
    ``"sin"`` (1 input). The function's output is the value of the LAST
    node in the tape.

    Return the gradient of the output with respect to every entry of
    ``x``, as a 1-D array of shape ``(len(x),)``, computed via a forward
    pass followed by a reverse-mode (adjoint) backward pass.
    """
    raise NotImplementedError('your code here')
