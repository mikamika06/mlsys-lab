import numpy as np

def predict_overflow(logits_list):
    thresh = np.log(np.finfo(np.float32).max)
    return [log.max() > thresh for log in logits_list]
