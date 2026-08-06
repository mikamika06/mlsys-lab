import hashlib

def block_hashes(tokens, block_size):
    prev = b""
    out = []
    for i in range(0, (len(tokens) // block_size) * block_size, block_size):
        chunk = tokens[i:i+block_size]
        s = ",".join(map(str, chunk)).encode("utf-8")
        prev = hashlib.sha256(prev + s).digest()
        out.append(prev.hex())
    return out

def reusable_blocks(hashes1, hashes2):
    count = 0
    for h1, h2 in zip(hashes1, hashes2):
        if h1 == h2:
            count += 1
        else:
            break
    return count

def divergence(tokens1, tokens2, block_size):
    div_idx = 0
    for t1, t2 in zip(tokens1, tokens2):
        if t1 == t2:
            div_idx += 1
        else:
            break

    lost = 0
    limit = min(len(tokens1), len(tokens2))
    for i in range(0, (limit // block_size) * block_size, block_size):
        chunk1 = tokens1[i:i+block_size]
        chunk2 = tokens2[i:i+block_size]
        if chunk1 == chunk2 and (i + block_size > div_idx):
            lost += 1

    return div_idx, lost
