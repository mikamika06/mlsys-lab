import numpy as np

def analyze_model_mixing():
    rng = np.random.default_rng(42)
    model_a_embeds = rng.normal(size=(5, 64))
    model_b_embeds = rng.normal(size=(5, 64)) + 10.0

    sim_cross = np.dot(model_a_embeds[0], model_b_embeds[0]) / (
        np.linalg.norm(model_a_embeds[0]) * np.linalg.norm(model_b_embeds[0])
    )
    return {"status": "analyzed", "cross_similarity": float(sim_cross)}
