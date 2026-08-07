import hashlib


def compute_block_hash(block_data, is_image=False, truncate_bits=32):
    prefix = b"IMG:" if is_image else b"TXT:"
    hasher = hashlib.sha256()
    hasher.update(prefix)
    if isinstance(block_data, str):
        hasher.update(block_data.encode("utf-8"))
    else:
        hasher.update(block_data)
    full_digest = int.from_bytes(hasher.digest(), byteorder="big")
    mask = (1 << truncate_bits) - 1
    return full_digest & mask


def search_collision(blocks, truncate_bits=32):
    seen = {}
    collisions = []
    for idx, (b_data, is_img) in enumerate(blocks):
        h = compute_block_hash(b_data, is_image=is_img, truncate_bits=truncate_bits)
        if h in seen:
            collisions.append((seen[h], idx, h))
        else:
            seen[h] = idx
    return collisions
