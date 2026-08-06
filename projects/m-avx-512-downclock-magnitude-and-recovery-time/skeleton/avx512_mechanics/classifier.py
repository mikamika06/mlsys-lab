def classify_instruction(instr: str) -> str:
    """Classify an assembly instruction into ISA tier ('L0', 'L1', 'L2')."""
    raise NotImplementedError


def classify_snippet(instructions: list[str]) -> str:
    """Return the highest ISA tier present in a sequence of instructions."""
    raise NotImplementedError
