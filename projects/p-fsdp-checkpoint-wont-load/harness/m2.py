import ref


def check(workdir):
    import fsdp_ckpt.parser as parser

    m = {"aligned": 0.0, "correct_unpadded_len": 0.0}
    ckpt = ref.get_fixture_ckpt(4)
    chunks = ref.extract_chunks(ckpt)
    metadata = ref.get_metadata()

    out = parser.align_shapes(chunks, metadata)
    expected = ref.align_shapes(chunks, metadata)

    if out.keys() == expected.keys():
        m["aligned"] = 1.0
        lengths_match = True
        for k in out:
            if out[k][1] != expected[k][1]:
                lengths_match = False
        if lengths_match:
            m["correct_unpadded_len"] = 1.0

    return m
