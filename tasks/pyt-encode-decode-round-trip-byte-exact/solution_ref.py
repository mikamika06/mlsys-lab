def encode_decode_round_trip(strings):
    result = {}
    for encoding in ["utf-8", "utf-16-le", "latin-1"]:
        items = []
        for s in strings:
            encoded = s.encode(encoding)
            decoded = encoded.decode(encoding)
            items.append((encoded, decoded))
        result[encoding] = items
    return result
