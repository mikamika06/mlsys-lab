import numpy as np


def padding_waste_fraction(lens: np.ndarray, batch_ids: np.ndarray) -> float:
    """
    lens      : 1-D int array, total (prompt + generation) token length of
                each request.
    batch_ids : 1-D int array, same length as `lens`; the STATIC batch each
                request was assigned to.

    Return the fraction of allocated token slots that are pure padding
    waste, summed over all batches:
        sum_b [max(L_b)*|L_b| - sum(L_b)] / sum_b [max(L_b)*|L_b|]
    """
    raise NotImplementedError('your code here')
