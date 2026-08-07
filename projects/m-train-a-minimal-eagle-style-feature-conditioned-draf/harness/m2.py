import numpy as np
import ref


def check(workdir):
    from eagledraft.model import TokenOnlyDraftHead, EagleFeatureDraftHead
    from eagledraft.eval import compare_draft_heads

    out = {"feature_advantage": 0.0}

    token_ids, hidden_states, target_tokens = ref.get_synthetic_dataset(seed=100)

    t_head = TokenOnlyDraftHead(
        ref.VOCAB_SIZE, ref.EMBED_DIM, seed=42
    )
    e_head = EagleFeatureDraftHead(
        ref.VOCAB_SIZE, ref.EMBED_DIM, ref.HIDDEN_DIM, seed=42
    )

    e_head.proj_feat[0, :] = 5.0
    e_head.fc[:ref.EMBED_DIM, :] = 0.0
    e_head.fc[ref.EMBED_DIM:, :] = 5.0
    e_head.head[:, 0] = -10.0
    e_head.head[:, 1] = 10.0

    try:
        res = compare_draft_heads(
            t_head, e_head, token_ids, hidden_states, target_tokens
        )
    except Exception as e:
        out["_note"] = f"compare_draft_heads failed: {type(e).__name__}: {str(e)}"
        return out

    e_acc = res.get("eagle_acc", 0.0)
    t_acc = res.get("token_acc", 0.0)
    diff = e_acc - t_acc

    out["feature_advantage"] = float(diff)
    return out
