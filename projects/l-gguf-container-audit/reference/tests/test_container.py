from ggufkit import container, plan


def assert_loadable(blob):
    head = container.parse_header(blob)
    assert head["magic"] == "GGUF", "not a GGUF file: %r" % head["magic"]
    assert head["version"] == 3, "unsupported version %d" % head["version"]

    index = container.parse_tensor_index(blob)
    assert index["tensors"], "no tensors in index"

    problems = container.validate(blob)
    assert not problems, "; ".join(problems)

    for t in index["tensors"]:
        end = t["absolute_data_offset"] + t["n_bytes"]
        assert end <= len(blob), (
            "%s: data ends at %d, file is %d bytes" % (t["name"], end, len(blob)))

    p = plan.load_plan(blob)
    assert p["resident_bytes"] >= p["weight_bytes"], (
        "load plan faults in fewer bytes than the weights occupy")


def test_clean(blob):
    assert_loadable(blob)


def test_corrupt(blob):
    try:
        assert_loadable(blob)
    except AssertionError:
        return
    raise AssertionError("damaged container was accepted")
