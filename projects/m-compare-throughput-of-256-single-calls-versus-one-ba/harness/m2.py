import ref


def check(workdir):
    from embedrun.analysis import check_l2_normalized, analyze_model_mixing

    embs = ref.get_reference_embeddings()
    norm_ok = check_l2_normalized(embs)

    rng = np.random.default_rng(999)
    emb_a = rng.normal(size=(10, 32))
    emb_a /= np.linalg.norm(emb_a, axis=-1, keepdims=True)
    emb_b = rng.normal(size=(10, 32)) * 5.0

    analysis = analyze_model_mixing(emb_a, emb_b)
    mixing_detected = 1.0 if not analysis["compatible"] else 0.0

    out = {
        "norm_detected": 1.0 if norm_ok else 0.0,
        "mixing_failure_detected": mixing_detected
    }
    return out
