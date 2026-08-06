def verify_salt_isolation(tenant_a_blocks, tenant_b_blocks, salt_a, salt_b):
    hashed_a = {hash((b, salt_a)) for b in tenant_a_blocks}
    hashed_b = {hash((b, salt_b)) for b in tenant_b_blocks}
    return len(hashed_a.intersection(hashed_b)) == 0
