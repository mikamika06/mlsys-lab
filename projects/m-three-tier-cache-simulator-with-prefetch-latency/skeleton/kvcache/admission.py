class AdmissionPolicy:
    """Base class for cache admission policies."""

    def should_admit(self, key: str, size_bytes: int, access_count: int, tier_free_bytes: int) -> bool:
        raise NotImplementedError


class AlwaysAdmit(AdmissionPolicy):
    def should_admit(self, key: str, size_bytes: int, access_count: int, tier_free_bytes: int) -> bool:
        raise NotImplementedError


class ReuseCountAdmit(AdmissionPolicy):
    def __init__(self, min_reuse: int = 2):
        raise NotImplementedError

    def should_admit(self, key: str, size_bytes: int, access_count: int, tier_free_bytes: int) -> bool:
        raise NotImplementedError


class SizeAwareAdmit(AdmissionPolicy):
    def __init__(self, max_size_bytes: int):
        raise NotImplementedError

    def should_admit(self, key: str, size_bytes: int, access_count: int, tier_free_bytes: int) -> bool:
        raise NotImplementedError
