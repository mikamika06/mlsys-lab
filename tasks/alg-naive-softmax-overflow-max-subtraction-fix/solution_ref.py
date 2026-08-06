import math

def softmax(logits: list[float]) -> list[float]:
    if isinstance(logits, list):
        if len(logits) > 0 and isinstance(logits[0], list):
            return [softmax(row) for row in logits]
        else:
            last_dim = len(logits)
            if last_dim == 0:
                return []
            m = float(logits[0])
            for j in range(1, last_dim):
                v = float(logits[j])
                if v > m:
                    m = v
            exps = [0.0] * last_dim
            s = 0.0
            for j in range(last_dim):
                e = math.exp(float(logits[j]) - m)
                exps[j] = e
                s += e
            out = [0.0] * last_dim
            for j in range(last_dim):
                out[j] = exps[j] / s
            return out
    return logits
