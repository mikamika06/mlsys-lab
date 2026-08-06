def test_partitioner_does_not_group_unsupported():
    from delegate_measure.partitioner import partition_xnnpack

    ops = [
        {"opcode": "CONV_2D", "flops": 100.0},
        {"opcode": "UNSUPPORTED_OP", "flops": 50.0},
        {"opcode": "ADD", "flops": 10.0}
    ]

    res = partition_xnnpack(ops)
    if len(res) != 3:
        raise ValueError("Partitioner incorrectly grouped unsupported operations together")
