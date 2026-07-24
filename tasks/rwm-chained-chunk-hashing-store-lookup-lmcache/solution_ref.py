import hashlib


def lookup_reused_chunks(chunks: list[bytes], store: dict[bytes, list[int]]) -> list[tuple[int, int]]:
    previous = b""
    result = set()

    for index, chunk in enumerate(chunks):
        previous = hashlib.sha256(previous + chunk).digest()
        if previous in store:
            for position in store[previous]:
                result.add((index, position))

    return sorted(result)
