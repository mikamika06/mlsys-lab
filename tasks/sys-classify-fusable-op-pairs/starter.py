def classify_fusable(op_pairs):
    """
    For each (producer_op_name, consumer_op_name) pair in `op_pairs`,
    return True if the producer op can be fused directly into the
    consumer op's kernel, per the op-kind rules in task.md, else False.

    Returns a list of bool, one per input pair, in order.
    """
    raise NotImplementedError('your code here')
