from partitioner.predict import partition_ops

def test_partitioning_disjoint_blobs():
    ops = [
        {'id': 0, 'type': 'Conv2D'},
        {'id': 1, 'type': 'Cast'},
        {'id': 2, 'type': 'Conv2D'}
    ]
    allowlist = {'Conv2D'}
    got = partition_ops(ops, allowlist)
    assert got == [0, -1, 1], f"Expected separated ops to have distinct blob IDs [0, -1, 1], got {got}"
