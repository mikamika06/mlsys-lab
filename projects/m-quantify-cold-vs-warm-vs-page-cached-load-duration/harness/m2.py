import ref

def check(workdir):
    from keepalive.controller import KeepAliveController
    ctrl_ref = ref.KeepAliveController(2000.0)
    ctrl_got = KeepAliveController(2000.0)

    traces = [
        [{"id": "a", "size_mb": 800, "priority": 10}, {"id": "b", "size_mb": 1000, "priority": 5}],
        [{"id": "a", "size_mb": 800, "priority": 10}, {"id": "c", "size_mb": 1500, "priority": 20}],
        [{"id": "b", "size_mb": 1000, "priority": 5}, {"id": "c", "size_mb": 1500, "priority": 20}]
    ]

    match = 1.0
    for trace in traces:
        r_out = sorted(ctrl_ref.update(trace))
        g_out = sorted(ctrl_got.update(trace))
        if r_out != g_out:
            match = 0.0
            break
    return {"controller_match": float(match)}
