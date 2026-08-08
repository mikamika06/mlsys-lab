import numpy as np
import ref


def check(workdir):
    import fsdp_ckpt.loss as loss_mod

    m = {"loss_matches": 0.0, "loss_invariant": 0.0}
    consolidated = ref.get_consolidated()
    meta = ref.get_metadata()
    inputs = {k: np.ones_like(consolidated[k]) * 0.1 for k in consolidated}

    ckpt4 = ref.shard_checkpoint(consolidated, 4)
    ckpt7 = ref.shard_checkpoint(consolidated, 7)

    val4 = loss_mod.compute_sharded_loss(ckpt4, meta, inputs)
    val7 = loss_mod.compute_sharded_loss(ckpt7, meta, inputs)
    expected = ref.compute_sharded_loss(ckpt4, meta, inputs)

    if abs(val4 - expected) < 1e-5:
        m["loss_matches"] = 1.0
    if abs(val4 - val7) < 1e-5:
        m["loss_invariant"] = 1.0

    return m
