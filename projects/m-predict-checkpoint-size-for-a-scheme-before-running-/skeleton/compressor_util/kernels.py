def get_supported_architectures(scheme_name: str) -> list:
    raise NotImplementedError


def has_native_kernel(scheme_name: str, arch: str) -> bool:
    raise NotImplementedError
