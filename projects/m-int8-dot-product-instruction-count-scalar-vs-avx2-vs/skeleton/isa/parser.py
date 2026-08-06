def analyze_objdump(lines: list[str]) -> dict:
    """
    Counts occurences of exact instruction mnemonics.
    Returns dict with keys: 'vpmaddubsw', 'vpmaddwd', 'vpaddd', 'vpdpbusd'
    """
    raise NotImplementedError
