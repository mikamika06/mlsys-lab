import numpy as np
import ref


def check(workdir):
    import fsdp_ckpt.converter as converter

    m = {"sharded_len_correct": 0.0, "padding_correct": 0.0, "num_ranks_handled": 0.0}
    consolidated = ref.get_consolidated()

    out = converter.shard_checkpoint(consolidated, 5)
    expected = ref.shard_checkpoint(consolidated, 5)

    if len(out) == 5:
        m["num_ranks_handled"] = 1.0
        lens_ok = True
        pads_ok = True
        for r in range(5):
            for k in expected[r]:
                if out[r][k].shape != expected[r][k].shape:
                    lens_ok = False
                if not np.array_equal(out[r][k], expected[r][k]):
                    pads_ok = False
        if lens_ok:
            m["sharded_len_correct"] = 1.0
        if pads_ok:
            m["padding_correct"] = 1.0

    return m
