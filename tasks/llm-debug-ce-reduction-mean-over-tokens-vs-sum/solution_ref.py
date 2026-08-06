import math

def cross_entropy_loss(
    logits: list[list[list[float]]],
    targets: list[list[int]],
    mask: list[list[bool]] | None = None,
) -> list[float]:
    losses = []
    for i in range(len(logits)):
        seq_logits = logits[i]
        seq_targets = targets[i]
        seq_mask = mask[i] if mask is not None else None

        total_ce = 0.0
        denom = 0

        for j in range(len(seq_logits)):
            if seq_mask is not None and not seq_mask[j]:
                continue

            tok_logits = seq_logits[j]
            t = seq_targets[j]

            max_val = max(tok_logits)
            exp_sum = sum(math.exp(v - max_val) for v in tok_logits)
            log_prob = (tok_logits[t] - max_val) - math.log(exp_sum + 1e-12)

            total_ce += -log_prob
            denom += 1

        if seq_mask is not None:
            if denom > 0:
                losses.append(total_ce / denom)
            else:
                losses.append(0.0)
        else:
            losses.append(total_ce / len(seq_logits))

    return losses
