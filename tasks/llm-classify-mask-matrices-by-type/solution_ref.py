def classify_masks(masks: list[list[list[bool]]]) -> list[str]:
    def classify(mask: list[list[bool]]) -> str:
        n = len(mask)

        is_all = True
        for i in range(n):
            for j in range(n):
                if not bool(mask[i][j]):
                    is_all = False
                    break
            if not is_all:
                break
        if is_all:
            return "bidirectional"

        is_causal = True
        for i in range(n):
            for j in range(n):
                expected = i >= j
                if bool(mask[i][j]) != expected:
                    is_causal = False
                    break
            if not is_causal:
                break
        if is_causal:
            return "causal"

        for w in range(1, n - 1):
            is_window = True
            for i in range(n):
                for j in range(n):
                    expected = (i >= j) and (i - j <= w)
                    if bool(mask[i][j]) != expected:
                        is_window = False
                        break
                if not is_window:
                    break
            if is_window:
                return "window"

        for p in range(1, n):
            is_prefix = True
            for i in range(n):
                for j in range(n):
                    expected = (i >= j) or (i < p and j < p)
                    if bool(mask[i][j]) != expected:
                        is_prefix = False
                        break
                if not is_prefix:
                    break
            if is_prefix:
                return "prefix-lm"

        return "unknown"

    return [classify(m) for m in masks]
