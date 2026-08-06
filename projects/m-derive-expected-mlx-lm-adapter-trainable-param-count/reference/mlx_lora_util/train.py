def simulate_training_loss(steps, initial_loss):
    losses = []
    curr = float(initial_loss)
    for i in range(steps):
        curr = max(0.1, curr * 0.95 - 0.01 * i * 0.001)
        losses.append(curr)
    return losses
