import ref
from tokenizer.converter import TokenizerConverter


def check(workdir):
    m = {"classes_identified": 0.0}
    conv = TokenizerConverter({}, [])
    res = conv.convert()
    if isinstance(res, dict):
        m["classes_identified"] = 1.0
    return m
