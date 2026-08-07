def mined_vocab_align(teacher_vocab: list[str], student_vocab: list[str]) -> list[int]:
    """
    For each token in teacher_vocab, return the index of the token in
    student_vocab with minimum Levenshtein edit distance (ties -> the
    smallest student index). See task.md for the edit-distance
    definition.
    """
    raise NotImplementedError('your code here')
