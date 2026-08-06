import ref


def check(workdir):
    from mlpackage.bytes import attribute_bytes
    tmp = ref.create_mock_package()
    try:
        want = ref.ref_attribute(tmp.name)
        got = attribute_bytes(tmp.name)
        if got == want:
            return {"bytes_match": 1.0}
        return {"bytes_match": 0.0, "_note": f"got {got}, want {want}"}
    finally:
        tmp.cleanup()
