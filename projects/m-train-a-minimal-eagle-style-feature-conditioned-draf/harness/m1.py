import numpy as np
import ref


def check(workdir):
    from eagledraft.model import EagleFeatureDraftHead

    out = {"rel_err": 1.0}

    token_ids, hidden_states, _ = ref.get_synthetic_dataset(seed=42)

    e_head = EagleFeatureDraftHead(
        vocab_size=ref.VOCAB_SIZE,
        embed_dim=ref.EMBED_DIM,
        hidden_dim=ref.HIDDEN_DIM,
        seed=42,
    )

    want_logits = ref.ref_eagle_forward(e_head, token_ids, hidden_states)

    try:
        got_logits = e_head.forward(token_ids, hidden_states)
    except Exception as e:
        out["_note"] = f"forward failed: {type(e).__name__}: {str(e)}"
        return out

    if got_logits is None or not isinstance(got_logits, np.ndarray):
        out["_note"] = "forward returned None or non-ndarray"
        return out

    diff = np.linalg.norm(want_logits - got_logits)
    norm = np.linalg.norm(want_logits) + 1e-8
    out["rel_err"] = float(diff / norm)
    return out
