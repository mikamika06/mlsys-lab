import re


def match_tensors(regex_pattern, tensor_names):
    pat = re.compile(regex_pattern)
    return [name for name in tensor_names if pat.search(name)]
