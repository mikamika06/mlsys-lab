class AdmissionPolicy:
    """Base class for cache admission policies."""

    def should_admit(self, key: str, size_bytes: int, access_count: int, tier_free_bytes: int) -> bool:
        return True


class AlwaysAdmit(AdmissionPolicy):
    def should_admit(self, key: str, size_bytes: int, access_count: int, tier_free_bytes: int) -> bool:
        return True


class ReuseCountAdmit(AdmissionPolicy):
    def __init__(self, min_reuse: int = 2):
        self.min_reuse = min_reuse

    def should_admit(self, key: str, size_bytes: int, access_count: int, tier_free_bytes: int) -> bool:
        return access_count >= self.min_reuse


class SizeAwareAdmit(AdmissionPolicy):
    def __init__(self, max_size_bytes: int):
        self.max_size_bytes = max_size_bytes

    def should_admit(self, key: str, size_bytes: int, access_count: int, tier_free_bytes: int) -> bool:
        return size_bytes <= self.max_size_bytes
