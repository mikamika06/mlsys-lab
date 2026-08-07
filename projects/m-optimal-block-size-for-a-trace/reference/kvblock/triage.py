"""Corrupt block-table triage and repair."""


def triage_block_table(block_table, total_seq_len, block_size, max_valid_block_id):
    """Audit and repair a corrupt block table."""
    expected_blocks = (total_seq_len + block_size - 1) // block_size if total_seq_len > 0 else 0
    repaired = list(block_table[:expected_blocks])

    issues = []

    if len(block_table) != expected_blocks:
        issues.append("length_mismatch")
        while len(repaired) < expected_blocks:
            repaired.append(-1)

    seen = set()
    for i in range(len(repaired)):
        blk_id = repaired[i]
        if blk_id < 0 or blk_id > max_valid_block_id or blk_id in seen:
            issues.append(f"invalid_or_duplicate_block_{i}")
            repaired[i] = -1
        else:
            seen.add(blk_id)

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "expected_blocks": expected_blocks,
        "repaired_table": repaired,
    }
