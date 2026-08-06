class CacheTracker:
    """Tracks block ownership and lookup events across requests."""

    def __init__(self, block_size=16):
        self.block_size = block_size
        self.block_owners = {}
        self.block_tokens = {}

    def process_event(self, event):
        event_type = event.get("type")
        tenant_id = event.get("tenant_id")
        block_id = event.get("block_id")

        if event_type == "allocate":
            tokens = event.get("tokens", [])
            self.block_owners[block_id] = tenant_id
            self.block_tokens[block_id] = list(tokens)
            return None

        if event_type == "lookup":
            owner = self.block_owners.get(block_id)
            if owner is not None and owner != tenant_id:
                tokens = self.block_tokens.get(block_id, [])
                return {
                    "request_id": event.get("request_id"),
                    "tenant_id": tenant_id,
                    "owner_tenant_id": owner,
                    "block_id": block_id,
                    "tokens_leaked": len(tokens),
                }
        return None

    def get_block_owners(self):
        return dict(self.block_owners)
