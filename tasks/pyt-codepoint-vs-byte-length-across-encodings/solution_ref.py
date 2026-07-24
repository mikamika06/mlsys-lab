def encoding_lengths(strings):
    result = []
    for s in strings:
        result.append(
            (
                len(s),
                len(s.encode("utf-8")),
                len(s.encode("utf-16-le")),
            )
        )
    return result
