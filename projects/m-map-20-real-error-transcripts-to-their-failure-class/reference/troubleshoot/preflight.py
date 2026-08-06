import ref


def can_fit(model_size_b, ctx, tp, vram_gb):
    return ref.predict_fit(model_size_b, ctx, tp, vram_gb)
