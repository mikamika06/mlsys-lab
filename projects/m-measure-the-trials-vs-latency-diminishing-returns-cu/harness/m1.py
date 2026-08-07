import ref


def check(workdir):
    from metatune.tasks import extract_tasks

    module_spec = ["matmul", "relu", "bias_add"]
    expected = ref.compute_reference_tasks(module_spec)
    try:
        got = extract_tasks(module_spec)
    except Exception as e:
        return {"tasks_matched": 0.0, "_note": f"extract_tasks raised {type(e).__name__}"}

    if not isinstance(got, list) or len(got) != len(expected):
        return {"tasks_matched": 0.0, "_note": f"expected {len(expected)} tasks, got {len(got) if isinstance(got, list) else type(got)}"}

    match = 1.0
    for g, e in zip(got, expected):
        if g.get("task_name") != e["task_name"]:
            match = 0.0
            break

    return {"tasks_matched": match}
