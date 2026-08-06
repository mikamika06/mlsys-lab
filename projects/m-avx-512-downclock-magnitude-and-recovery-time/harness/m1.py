import ref


def check(workdir):
    from avx512_mechanics.classifier import classify_instruction, classify_snippet

    out = {"tier_matches": 0.0}
    cases = ref.get_test_snippets()
    ok = 0
    total = len(cases)

    for instructions, expected_tier in cases:
        snip_tier = classify_snippet(instructions)
        if snip_tier == expected_tier:
            ok += 1
        elif "_note" not in out:
            out["_note"] = (
                f"Expected {expected_tier} for {instructions}, got {snip_tier}"
            )

    out["tier_matches"] = float(ok / total) if total > 0 else 0.0
    return out
