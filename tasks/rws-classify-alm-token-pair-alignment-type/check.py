def _oracle(teacher_token: str, student_token: str) -> str:
    """Classification oracle — the reference algorithm itself."""
    if teacher_token == student_token:
        return "exact"
    if teacher_token in student_token or student_token in teacher_token:
        return "substring"
    return "none"

def grade(sol, fx) -> dict:
    test_pairs = [
        # --- exact matches ---
        ("hello", "hello"),
        ("world", "world"),
        ("the", "the"),
        # --- substring: teacher ⊂ student ---
        ("hel", "hello"),
        ("wor", "world"),
        ("th", "the"),
        # --- substring: student ⊂ teacher (reversed) ---
        ("hello", "hel"),
        ("world", "wor"),
        ("python", "py"),
        # --- no match ---
        ("hello", "world"),
        ("cat", "dog"),
        ("foo", "bar"),
        # --- edge cases ---
        ("", ""),
        ("a", "a"),
        ("ab", "abc"),
        ("abc", "ab"),
        ("ca", "cat"),
        ("tion", "nation"),
        ("NLP", "nlp"),
        ("test", "testing"),
        ("$$", "$$$"),
        ("token", "tok"),
    ]

    # Compute reference answers with the oracle
    expected = [_oracle(t, s) for t, s in test_pairs]

    # Get learner's answers
    try:
        got = sol.classify_alignment(test_pairs)
    except Exception:
        return {"exact_match": 0.0}

    # Validate output shape
    if not isinstance(got, list) or len(got) != len(expected):
        return {"exact_match": 0.0}

    # Compare
    correct = sum(1 for g, e in zip(got, expected) if g == e)
    return {"exact_match": correct / len(expected)}
