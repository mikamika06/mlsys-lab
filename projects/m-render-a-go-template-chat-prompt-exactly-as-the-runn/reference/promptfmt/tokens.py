def find_mismatched_token(modelfile_meta, card_meta):
    s1 = set(modelfile_meta.get("special_tokens", []))
    if "eos_token" in modelfile_meta:
        s1.add(modelfile_meta["eos_token"])
    if "bos_token" in modelfile_meta:
        s1.add(modelfile_meta["bos_token"])

    s2 = set(card_meta.get("special_tokens", []))
    if "eos_token" in card_meta:
        s2.add(card_meta["eos_token"])
    if "bos_token" in card_meta:
        s2.add(card_meta["bos_token"])

    diff = s2 - s1
    if diff:
        return sorted(list(diff))[0]
    diff_rev = s1 - s2
    if diff_rev:
        return sorted(list(diff_rev))[0]
    return None
