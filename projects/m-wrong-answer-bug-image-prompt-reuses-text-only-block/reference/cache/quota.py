class TenantQuotaSimulator:
    def __init__(self, total_quota):
        self.total_quota = total_quota
        self.tenant_usage = {}
        self.block_owners = {}

    def allocate(self, tenant_id, block_hash):
        current_total = sum(self.tenant_usage.values())
        if block_hash in self.block_owners:
            owner = self.block_owners[block_hash]
            if owner == tenant_id:
                return True
            return False

        if current_total >= self.total_quota:
            return False

        self.block_owners[block_hash] = tenant_id
        self.tenant_usage[tenant_id] = self.tenant_usage.get(tenant_id, 0) + 1
        return True

    def free(self, tenant_id, block_hash):
        if self.block_owners.get(block_hash) == tenant_id:
            del self.block_owners[block_hash]
            self.tenant_usage[tenant_id] -= 1
            if self.tenant_usage[tenant_id] <= 0:
                del self.tenant_usage[tenant_id]
            return True
        return False
