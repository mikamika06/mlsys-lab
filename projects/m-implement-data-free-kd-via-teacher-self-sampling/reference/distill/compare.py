import numpy as np


def evaluate_distillation(synthetic_logits, real_logits, true_labels):
    synth_pred = np.argmax(synthetic_logits, axis=-1)
    real_pred = np.argmax(real_logits, axis=-1)
    synth_acc = float(np.mean(synth_pred == true_labels))
    real_acc = float(np.mean(real_pred == true_labels))
    delta = float(np.abs(synth_acc - real_acc))
    return delta, synth_acc, real_acc
