def validate_fix(sequence):
    ops = list(sequence)
    if "scatter" in ops and "gather" in ops:
        if ops.index("scatter") < ops.index("gather"):
            return True
    return False
