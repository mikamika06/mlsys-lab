import re

SHARD_REGEX = re.compile(r"^(.+)-(\d{5})-of-(\d{5})\.gguf$")

def parse_shard_filename(filename):
    m = SHARD_REGEX.match(filename)
    if not m:
        return None
    prefix, idx, total = m.groups()
    return prefix, int(idx), int(total)

def validate_shard_set(filenames):
    if not filenames:
        return False, "empty shard set"
    parsed = []
    prefix = None
    total_expected = None
    for fn in filenames:
        res = parse_shard_filename(fn)
        if not res:
            return False, f"invalid filename format: {fn}"
        p, idx, total = res
        if prefix is None:
            prefix = p
            total_expected = total
        else:
            if p != prefix or total != total_expected:
                return False, "inconsistent prefix or total count"
        parsed.append(idx)

    parsed.sort()
    if parsed[0] != 1 or parsed[-1] != total_expected:
        return False, "sequence bounds mismatch"
    if parsed != list(range(1, total_expected + 1)):
        return False, "missing or duplicate shards in sequence"
    return True, "ok"
