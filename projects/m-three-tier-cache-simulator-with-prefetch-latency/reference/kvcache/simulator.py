class ThreeTierCache:
    """Three-tier cache simulator with prefetch latency."""

    def __init__(self, vram_cap: int, dram_cap: int, nvme_cap: int,
                 vram_lat: float = 1.0, dram_lat: float = 10.0, nvme_lat: float = 100.0,
                 dram_bw_gbps: float = 50.0, nvme_bw_gbps: float = 10.0):
        self.caps = {'vram': vram_cap, 'dram': dram_cap, 'nvme': nvme_cap}
        self.lats = {'vram': vram_lat, 'dram': dram_lat, 'nvme': nvme_lat}
        self.bws = {'dram': dram_bw_gbps, 'nvme': nvme_bw_gbps}
        self.stores = {'vram': {}, 'dram': {}, 'nvme': {}}
        self.used = {'vram': 0, 'dram': 0, 'nvme': 0}
        self.lru = {'vram': [], 'dram': [], 'nvme': []}
        self.prefetch_queue = {}

    def _touch_lru(self, tier: str, key: str):
        if key in self.lru[tier]:
            self.lru[tier].remove(key)
        self.lru[tier].append(key)

    def _evict(self, tier: str, needed_bytes: int):
        while self.caps[tier] - self.used[tier] < needed_bytes and self.lru[tier]:
            victim = self.lru[tier].pop(0)
            sz = self.stores[tier].pop(victim)
            self.used[tier] -= sz

    def _place(self, tier: str, key: str, size_bytes: int):
        self._evict(tier, size_bytes)
        if self.caps[tier] - self.used[tier] >= size_bytes:
            self.stores[tier][key] = size_bytes
            self.used[tier] += size_bytes
            self._touch_lru(tier, key)
            return True
        return False

    def access(self, key: str, size_bytes: int, current_time: float, prefetch_keys: list[str] = None) -> dict:
        prefetch_keys = prefetch_keys or []
        for pkey in prefetch_keys:
            if pkey not in self.stores['vram'] and pkey not in self.prefetch_queue:
                source_tier = 'dram' if pkey in self.stores['dram'] else ('nvme' if pkey in self.stores['nvme'] else None)
                if source_tier:
                    transfer_time = (size_bytes / (self.bws[source_tier] * 1e9)) * 1e6
                    ready_at = current_time + self.lats[source_tier] + transfer_time
                    self.prefetch_queue[pkey] = (ready_at, source_tier, size_bytes)

        if key in self.stores['vram']:
            self._touch_lru('vram', key)
            return {'tier': 'vram', 'latency_us': self.lats['vram'], 'hit': True}

        if key in self.prefetch_queue:
            ready_at, src, sz = self.prefetch_queue.pop(key)
            if current_time >= ready_at:
                if key in self.stores[src]:
                    self.used[src] -= self.stores[src].pop(key)
                    if key in self.lru[src]:
                        self.lru[src].remove(key)
                self._place('vram', key, sz)
                return {'tier': 'vram', 'latency_us': self.lats['vram'], 'hit': True}

        for tier in ['dram', 'nvme']:
            if key in self.stores[tier]:
                self._touch_lru(tier, key)
                transfer_time = (size_bytes / (self.bws[tier] * 1e9)) * 1e6
                tot_lat = self.lats[tier] + transfer_time
                self.used[tier] -= self.stores[tier].pop(key)
                if key in self.lru[tier]:
                    self.lru[tier].remove(key)
                self._place('vram', key, size_bytes)
                return {'tier': tier, 'latency_us': tot_lat, 'hit': True}

        self._place('vram', key, size_bytes)
        return {'tier': 'miss', 'latency_us': self.lats['nvme'] * 2.0, 'hit': False}
