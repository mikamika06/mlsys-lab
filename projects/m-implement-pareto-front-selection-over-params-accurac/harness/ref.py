import numpy as np

SAMPLE_POINTS = [
    {"id": "c1", "params": 10, "accuracy": 0.70},
    {"id": "c2", "params": 20, "accuracy": 0.80},
    {"id": "c3", "params": 15, "accuracy": 0.75},
    {"id": "c4", "params": 30, "accuracy": 0.78},
]

def select_pareto_front(points):
    sorted_pts = sorted(points, key=lambda x: (x["params"], -x["accuracy"]))
    front = []
    best_acc = -float("inf")
    for p in sorted_pts:
        if p["accuracy"] > best_acc:
            front.append(p)
            best_acc = p["accuracy"]
    return front

STUDENT_LOGITS = np.array([[1.5, 0.5], [0.1, 2.0], [1.0, 1.0]])
TEACHER_LOGITS = np.array([[1.6, 0.4], [0.2, 1.9], [1.1, 0.9]])
TARGETS = np.array([0, 1, 0])

def measure_retention(student_logits, teacher_logits, targets):
    student_preds = np.argmax(student_logits, axis=-1)
    teacher_preds = np.argmax(teacher_logits, axis=-1)
    student_acc = np.mean(student_preds == targets)
    teacher_acc = np.mean(teacher_preds == targets)
    if teacher_acc == 0:
        return 0.0
    return float(student_acc / teacher_acc)

CHECKPOINTS = [
    {"id": "model_small", "params": 10, "accuracy": 0.75},
    {"id": "model_medium", "params": 25, "accuracy": 0.85},
    {"id": "model_large", "params": 50, "accuracy": 0.84},
]

def compare_checkpoints(checkpoints):
    points = [{"id": cp["id"], "params": cp["params"], "accuracy": cp["accuracy"]} for cp in checkpoints]
    front = select_pareto_front(points)
    front_ids = {p["id"] for p in front}
    results = []
    for cp in checkpoints:
        results.append({
            "id": cp["id"],
            "is_pareto": cp["id"] in front_ids,
            "params": cp["params"],
            "accuracy": cp["accuracy"]
        })
    return results
