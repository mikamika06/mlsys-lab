VALUE_TOKENS = {"STR", "NUM", "TRUE", "FALSE", "NULL"}
VALUE_START = VALUE_TOKENS | {"{", "["}


def _run_pda(prefix):
    """Advance a bracket-stack pushdown automaton over `prefix`.

    Each stack frame is one of:
      OBJ_START, OBJ_NEED_KEY, OBJ_NEED_COLON, OBJ_NEED_VALUE, OBJ_AFTER_VALUE,
      ARR_START, ARR_NEED_VALUE, ARR_AFTER_VALUE
    """
    stack = []
    done = False

    def complete_value():
        nonlocal done
        if not stack:
            done = True
            return
        top = stack[-1]
        if top == "OBJ_NEED_VALUE":
            stack[-1] = "OBJ_AFTER_VALUE"
        elif top in ("ARR_START", "ARR_NEED_VALUE"):
            stack[-1] = "ARR_AFTER_VALUE"
        else:
            raise ValueError(f"invalid prefix: value completed under {top}")

    for tok in prefix:
        if not stack:
            if done:
                raise ValueError("token after a complete top-level value")
            if tok in VALUE_TOKENS:
                complete_value()
            elif tok == "{":
                stack.append("OBJ_START")
            elif tok == "[":
                stack.append("ARR_START")
            else:
                raise ValueError(f"invalid top-level token {tok}")
            continue

        top = stack[-1]
        if top == "OBJ_START":
            if tok == "STR":
                stack[-1] = "OBJ_NEED_COLON"
            elif tok == "}":
                stack.pop()
                complete_value()
            else:
                raise ValueError(f"invalid token {tok} in OBJ_START")
        elif top == "OBJ_NEED_KEY":
            if tok == "STR":
                stack[-1] = "OBJ_NEED_COLON"
            else:
                raise ValueError(f"invalid token {tok} in OBJ_NEED_KEY")
        elif top == "OBJ_NEED_COLON":
            if tok == ":":
                stack[-1] = "OBJ_NEED_VALUE"
            else:
                raise ValueError(f"invalid token {tok} in OBJ_NEED_COLON")
        elif top == "OBJ_NEED_VALUE":
            if tok in VALUE_TOKENS:
                complete_value()
            elif tok == "{":
                stack.append("OBJ_START")
            elif tok == "[":
                stack.append("ARR_START")
            else:
                raise ValueError(f"invalid token {tok} in OBJ_NEED_VALUE")
        elif top == "OBJ_AFTER_VALUE":
            if tok == ",":
                stack[-1] = "OBJ_NEED_KEY"
            elif tok == "}":
                stack.pop()
                complete_value()
            else:
                raise ValueError(f"invalid token {tok} in OBJ_AFTER_VALUE")
        elif top == "ARR_START":
            if tok in VALUE_TOKENS:
                complete_value()
            elif tok == "{":
                stack.append("OBJ_START")
            elif tok == "[":
                stack.append("ARR_START")
            elif tok == "]":
                stack.pop()
                complete_value()
            else:
                raise ValueError(f"invalid token {tok} in ARR_START")
        elif top == "ARR_NEED_VALUE":
            if tok in VALUE_TOKENS:
                complete_value()
            elif tok == "{":
                stack.append("OBJ_START")
            elif tok == "[":
                stack.append("ARR_START")
            else:
                raise ValueError(f"invalid token {tok} in ARR_NEED_VALUE")
        elif top == "ARR_AFTER_VALUE":
            if tok == ",":
                stack[-1] = "ARR_NEED_VALUE"
            elif tok == "]":
                stack.pop()
                complete_value()
            else:
                raise ValueError(f"invalid token {tok} in ARR_AFTER_VALUE")
        else:
            raise ValueError(f"unknown frame {top}")

    return stack, done


def allowed_next_tokens(prefix: list) -> list:
    """Set of token types legally allowed immediately after `prefix`."""
    stack, done = _run_pda(prefix)

    if not stack:
        return [] if done else sorted(VALUE_START)

    top = stack[-1]
    if top == "OBJ_START":
        allowed = {"STR", "}"}
    elif top == "OBJ_NEED_KEY":
        allowed = {"STR"}
    elif top == "OBJ_NEED_COLON":
        allowed = {":"}
    elif top == "OBJ_NEED_VALUE":
        allowed = set(VALUE_START)
    elif top == "OBJ_AFTER_VALUE":
        allowed = {",", "}"}
    elif top == "ARR_START":
        allowed = set(VALUE_START) | {"]"}
    elif top == "ARR_NEED_VALUE":
        allowed = set(VALUE_START)
    elif top == "ARR_AFTER_VALUE":
        allowed = {",", "]"}
    else:
        raise ValueError(f"unknown frame {top}")

    return sorted(allowed)
