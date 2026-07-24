def classify_alignment(pairs):
    """Classify teacher-student token pairs as exact, substring, or none."""
    results = []
    for teacher_token, student_token in pairs:
        if teacher_token == student_token:
            results.append("exact")
        elif teacher_token in student_token or student_token in teacher_token:
            results.append("substring")
        else:
            results.append("none")
    return results
