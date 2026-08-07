import ref
from amp_fix.scaler_utils import analyze_scaler_step


def check(workdir):
    m = {"scaler_behavior_ok": 0.0}
    state = ref.get_mock_scaler_state()
    res_inf = analyze_scaler_step(state, has_inf=True)
    res_ok = analyze_scaler_step(state, has_inf=False)
    if res_inf["skipped"] is True and res_inf["scale"] < state["scale"] and res_ok["skipped"] is False and res_ok["scale"] > state["scale"]:
        m["scaler_behavior_ok"] = 1.0
    return m
