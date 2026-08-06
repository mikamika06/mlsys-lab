def verify_resume_state(checkpoint_state: dict, expected_step: int) -> bool:
    if not isinstance(checkpoint_state, dict):
        return False
    if checkpoint_state.get("step", -1) != expected_step:
        return False
    if "model_weights" not in checkpoint_state or not isinstance(checkpoint_state["model_weights"], dict):
        return False
    return True
