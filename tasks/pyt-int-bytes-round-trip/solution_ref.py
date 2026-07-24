def int_bytes_round_trip(values, width):
    result = []
    for value in values:
        encoded = int(value).to_bytes(width, byteorder="little", signed=False)
        decoded = int.from_bytes(encoded, byteorder="little", signed=False)
        if decoded != int(value):
            raise ValueError("round trip mismatch")
        result.append(encoded)
    return result
