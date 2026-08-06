def map_op_to_primitives(op_name: str) -> list[str]:
    """Maps a recorded framework op to Apple MPSGraph primitive name(s)."""
    raise NotImplementedError


def map_recorded_sequence(op_sequence: list[str]) -> list[str]:
    """Maps a sequence of recorded ops into a flattened list of MPSGraph primitives."""
    raise NotImplementedError
