import random


def generate_traces(seed=42):
    rng = random.Random(seed)
    tenants = [f"tenant_{i}" for i in range(5)]
    logs = []
    allocated_blocks = {}

    block_id_counter = 1000
    for req_idx in range(120):
        tenant = rng.choice(tenants)
        req_id = f"req_{req_idx}"

        if allocated_blocks and rng.random() < 0.6:
            target_block = rng.choice(list(allocated_blocks.keys()))
            logs.append({
                "type": "lookup",
                "request_id": req_id,
                "block_id": target_block,
                "tenant_id": tenant,
            })
        else:
            block_id = block_id_counter
            block_id_counter += 1
            tokens = [rng.randint(1, 50000) for _ in range(16)]
            allocated_blocks[block_id] = (tenant, tokens)
            logs.append({
                "type": "allocate",
                "block_id": block_id,
                "tenant_id": tenant,
                "tokens": tokens,
            })

    return logs
