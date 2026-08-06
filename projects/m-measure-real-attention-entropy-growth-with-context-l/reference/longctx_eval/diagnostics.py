import numpy as np

def diagnose_models(accuracies, max_entropies, lengths, acc_thresh, ent_thresh):
    out = []
    mean_x = np.mean(lengths)
    var_x = np.sum((lengths - mean_x)**2)
    num_models = accuracies.shape[0]

    for i in range(num_models):
        y = accuracies[i]
        mean_y = np.mean(y)
        slope = np.sum((lengths - mean_x) * (y - mean_y)) / var_x

        final_acc = accuracies[i, -1]
        final_ent = max_entropies[i, -1]

        if final_acc >= acc_thresh:
            mode = 'none'
        elif final_ent > ent_thresh:
            mode = 'dilution'
        else:
            mode = 'rope'

        out.append({
            'model_idx': i,
            'slope': float(slope),
            'mode': mode
        })

    return out
