import numpy as np


def evaluate_top1_accuracy(
    logits: np.ndarray, target_tokens: np.ndarray
) -> float:
    preds = np.argmax(logits, axis=-1)
    return float(np.mean(preds == target_tokens))


def compare_draft_heads(
    token_head, eagle_head, token_ids, hidden_states, target_tokens
):
    t_logits = token_head.predict_logits(token_ids)
    e_logits = eagle_head.forward(token_ids, hidden_states)

    t_acc = evaluate_top1_accuracy(t_logits, target_tokens)
    e_acc = evaluate_top1_accuracy(e_logits, target_tokens)

    return {"token_acc": t_acc, "eagle_acc": e_acc, "diff": e_acc - t_acc}
