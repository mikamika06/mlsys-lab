import torch
from zero1.formula import compute_zero1_memory
from zero1.toy_zero import partition_optimizer_states
from zero1.parser import parse_deepspeed_log


def oracle_compute_memory(num_params, world_size, bytes_per_elem=4, optimizer_type="adam"):
    return compute_zero1_memory(num_params, world_size, bytes_per_elem, optimizer_type)


def oracle_partition(params, world_size, rank):
    return partition_optimizer_states(params, world_size, rank)


def oracle_parse(log_text):
    return parse_deepspeed_log(log_text)
