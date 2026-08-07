import re


def build_cpu_moe_regex():
    return r"\.ffn_.*_exps.*=CPU"


def build_n_cpu_moe_regex(n):
    layers = "|".join(str(i) for i in range(n))
    return rf"^blk\.({layers})\.ffn_.*_exps.*=CPU"
