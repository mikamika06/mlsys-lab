import os

def get_safe_resume_checkpoint(checkpoint_dir: str) -> str:
    latest_path = os.path.join(checkpoint_dir, "latest")
    
    if os.path.isfile(latest_path):
        with open(latest_path, "r") as f:
            candidate = f.read().strip()
        candidate_path = os.path.join(checkpoint_dir, candidate)
        if os.path.isfile(os.path.join(candidate_path, "model.pt")):
            return candidate_path

    best_step = -1
    best_path = None
    if os.path.isdir(checkpoint_dir):
        for entry in os.listdir(checkpoint_dir):
            if entry.startswith("step_"):
                try:
                    step = int(entry.split("_")[1])
                    path = os.path.join(checkpoint_dir, entry)
                    if os.path.isfile(os.path.join(path, "model.pt")):
                        if step > best_step:
                            best_step = step
                            best_path = path
                except ValueError:
                    pass
    return best_path
