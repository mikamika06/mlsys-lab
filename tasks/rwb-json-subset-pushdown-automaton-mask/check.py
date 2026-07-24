import numpy as np

TOKENS = ["{", "}", "[", "]", ":", ",", "STR", "NUM", "TRUE", "FALSE", "NULL"]
VALUE_TOKENS = {"STR", "NUM", "TRUE", "FALSE", "NULL"}
VALUE_START = VALUE_TOKENS | {"{", "["}


def _run_pda(prefix):
    """Advance the bracket-stack automaton over a (assumed valid) prefix."""
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


def _oracle_allowed(prefix):
    stack, done = _run_pda(prefix)
    if not stack:
        return set() if done else set(VALUE_START)
    top = stack[-1]
    if top == "OBJ_START":
        return {"STR", "}"}
    if top == "OBJ_NEED_KEY":
        return {"STR"}
    if top == "OBJ_NEED_COLON":
        return {":"}
    if top == "OBJ_NEED_VALUE":
        return set(VALUE_START)
    if top == "OBJ_AFTER_VALUE":
        return {",", "}"}
    if top == "ARR_START":
        return set(VALUE_START) | {"]"}
    if top == "ARR_NEED_VALUE":
        return set(VALUE_START)
    if top == "ARR_AFTER_VALUE":
        return {",", "]"}
    raise ValueError(f"unknown frame {top}")


def grade(sol, fx) -> dict:
    prefixes = np.asarray(fx["prefixes"], dtype=np.int64)
    lengths = np.asarray(fx["lengths"], dtype=np.int64)

    ok = 1.0
    for i in range(prefixes.shape[0]):
        L = int(lengths[i])
        prefix = [TOKENS[int(t)] for t in prefixes[i, :L]]
        expected = _oracle_allowed(prefix)
        try:
            got = sol.allowed_next_tokens(list(prefix))
            got_set = set(got)
        except Exception:
            ok = 0.0
            break
        if got_set != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
