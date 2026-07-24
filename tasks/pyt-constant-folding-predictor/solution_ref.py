import dis


def _folds_to_single_const(expr):
    """Compile `expr` as a lambda body and check whether the compiler's
    constant folder reduced it to loading one precomputed constant, instead
    of building the value at runtime."""
    code = compile("lambda: " + expr, "<predict_folded>", "eval")
    fn = eval(code)
    ops = [ins.opname for ins in dis.get_instructions(fn) if ins.opname != "RESUME"]
    # Python <3.12 compiles a folded constant as LOAD_CONST; RETURN_VALUE.
    # Python >=3.12 can compile it as a single RETURN_CONST instead.
    return ops == ["LOAD_CONST", "RETURN_VALUE"] or ops == ["RETURN_CONST"]


def predict_folded(exprs):
    """For each source expression string in `exprs`, return True if
    compiling it folds to a single constant, False otherwise."""
    return [_folds_to_single_const(e) for e in exprs]
