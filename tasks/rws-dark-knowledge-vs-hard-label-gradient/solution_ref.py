import math
import numpy as np


def _softmax(x):
    arr = np.asarray(x, dtype=np.float64)
    max_val = arr[0]
    for i in range(len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
    e = []
    total_sum = 0.0
    for i in range(len(arr)):
        val = math.exp(arr[i] - max_val)
        e.append(val)
        total_sum += val
    out = []
    for i in range(len(e)):
        out.append(e[i] / total_sum)
    return np.asarray(out, dtype=np.float64)


def kd_hard_label_grad(student_logits, teacher_probs, label, temperature):
    s_arr = np.asarray(student_logits, dtype=np.float64)
    t_arr = np.asarray(teacher_probs, dtype=np.float64)
    scaled = []
    for i in range(len(s_arr)):
        scaled.append(s_arr[i] / temperature)
    p_kd = _softmax(scaled)
    kd_grad_list = []
    for i in range(len(p_kd)):
        kd_grad_list.append(temperature * (p_kd[i] - t_arr[i]))
    kd_grad = np.asarray(kd_grad_list, dtype=np.float64)
    p_ce = _softmax(s_arr)
    ce_grad_list = []
    lbl = int(label)
    for i in range(len(p_ce)):
        y_val = 1.0 if i == lbl else 0.0
        ce_grad_list.append(p_ce[i] - y_val)
    ce_grad = np.asarray(ce_grad_list, dtype=np.float64)
    return kd_grad.astype(np.float64), ce_grad.astype(np.float64)
