class GGUFWriter:
    def __init__(self, alignment: int = 32) -> None:
        raise NotImplementedError

    def add_uint32(self, key: str, val: int) -> None:
        raise NotImplementedError

    def add_float32(self, key: str, val: float) -> None:
        raise NotImplementedError

    def add_string(self, key: str, val: str) -> None:
        raise NotImplementedError

    def add_tensor(self, name: str, shape: list[int], dtype_id: int, data: bytes) -> None:
        raise NotImplementedError

    def write(self) -> bytes:
        raise NotImplementedError
