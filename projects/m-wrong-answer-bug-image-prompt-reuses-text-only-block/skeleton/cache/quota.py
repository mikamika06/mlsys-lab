class TenantQuotaSimulator:
    def __init__(self, total_quota):
        raise NotImplementedError

    def allocate(self, tenant_id, block_hash):
        raise NotImplementedError

    def free(self, tenant_id, block_hash):
        raise NotImplementedError
