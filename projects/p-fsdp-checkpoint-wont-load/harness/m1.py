import numpy as np
import ref


def check(workdir):
    import fsdp_ckpt.parser as parser

    m = {"extracted": 0.0, "handles_empty": 0.0}
    ckpt = ref.get_fixture_ckpt(4)

    out = parser.extract_chunks(ckpt)
    expected = ref.extract_chunks(ckpt)

    if out.keys() == expected.keys():
        match = True
        for k in out:
            if len(out[k]) != 4:
                match = False
            for i in range(4):
                if not np.array_equal(out[k][i], expected[k][i]):
                    match = False
        if match:
            m["extracted"] = 1.0

    if parser.extract_chunks([]) == {}:
        m["handles_empty"] = 1.0

    return m
