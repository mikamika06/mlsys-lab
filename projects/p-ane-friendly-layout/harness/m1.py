import ref
from ane_model.audit import get_block_placement

def check(workdir):
    m = {"blocks_audited": 0.0}
    model = ref.get_sample_model()
    placement = get_block_placement(model)
    if isinstance(placement, dict) and len(placement) == 2:
        m["blocks_audited"] = 1.0
    return m
