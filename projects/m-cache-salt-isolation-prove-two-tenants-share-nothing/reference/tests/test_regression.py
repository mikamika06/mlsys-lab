from prefixhash.salt import verify_salt_isolation


def test_isolation():
    blocks = [1, 2, 3]
    assert verify_salt_isolation(blocks, blocks, "salt1", "salt1") is False
    assert verify_salt_isolation(blocks, blocks, "salt1", "salt2") is True
