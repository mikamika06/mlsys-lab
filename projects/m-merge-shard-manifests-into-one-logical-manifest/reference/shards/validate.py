from shards.manifest import validate_filename


def check_shard_set(filenames: list, manifests: list) -> bool:
    if len(filenames) != len(manifests):
        return False
    if not all(validate_filename(f) for f in filenames):
        return False
    totals = set()
    parts = set()
    for f in filenames:
        import re
        m = re.match(r"^(.+)-000([1-9][0-9]*)-of-000([1-9][0-9]*)\.gguf$", f)
        if not m:
            return False
        parts.add(int(m.group(2)))
        totals.add(int(m.group(3)))
    if len(totals) != 1:
        return False
    total = list(totals)[0]
    if total != len(filenames):
        return False
    if sorted(parts) != list(range(1, total + 1)):
        return False
    return True
