from zeroutil.comm import compute_zero_communication_volume
from zeroutil.logs import parse_deepspeed_init_log


def test_comm_non_negative():
    res = compute_zero_communication_volume(1024, 4)
    assert res["zero3_comm_bytes"] >= res["zero1_comm_bytes"]


def test_parse_init_log():
    lines = ["Rank 0: partition_size = 512", "Rank 1: partition_size = 512"]
    parsed = parse_deepspeed_init_log(lines)
    assert parsed[0] == 512
    assert parsed[1] == 512
