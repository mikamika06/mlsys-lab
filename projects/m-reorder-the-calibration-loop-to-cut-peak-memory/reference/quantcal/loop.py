def calibrate(model, dataloader, num_samples):
    processed = 0
    activations = {}
    for batch in dataloader:
        if processed >= num_samples:
            break
        out = model(batch)
        activations[processed] = out
        processed += 1
    return activations
