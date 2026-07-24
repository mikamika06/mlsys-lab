import numpy as np

def streaming_causal_softmax(logits, mask, block_size):
    """Broken implementation: ignores both the causal and boolean masks,
producing a full softmax over all logits.  This will fail the gate."""
    raise NotImplementedError('your code here')
