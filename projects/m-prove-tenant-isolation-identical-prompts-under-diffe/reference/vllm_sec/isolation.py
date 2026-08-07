import hashlib
from typing import List, Dict, Any, Tuple


def compute_block_hashes(tokens: List[int], block_size: int, tenant_salt: str) -> List[str]:
    hashes = []
    num_blocks = len(tokens) // block_size
    prefix_hash = ""
    for b in range(num_blocks):
        block_tokens = tokens[b * block_size: (b + 1) * block_size]
        payload = f"{prefix_hash}:{tenant_salt}:{','.join(map(str, block_tokens))}"
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        hashes.append(h)
        prefix_hash = h
    return hashes


def check_tenant_isolation(
    requests: List[Dict[str, Any]],
    block_size: int
) -> Tuple[bool, int]:
    tenant_blocks: Dict[str, set] = {}
    shared_count = 0

    for req in requests:
        tenant = req["tenant_id"]
        salt = req["tenant_salt"]
        tokens = req["tokens"]
        hashes = compute_block_hashes(tokens, block_size, salt)

        if tenant not in tenant_blocks:
            tenant_blocks[tenant] = set()

        for h in hashes:
            tenant_blocks[tenant].add(h)

    all_tenants = list(tenant_blocks.keys())
    for i in range(len(all_tenants)):
        for j in range(i + 1, len(all_tenants)):
            t1, t2 = all_tenants[i], all_tenants[j]
            overlap = tenant_blocks[t1].intersection(tenant_blocks[t2])
            shared_count += len(overlap)

    is_isolated = (shared_count == 0)
    return is_isolated, shared_count
