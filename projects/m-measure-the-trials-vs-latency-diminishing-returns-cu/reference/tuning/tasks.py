def extract_tasks(ir_module):
    extracted = []
    for f in ir_module.funcs:
        task_name = f"task_{f.name}"
        extracted.append({
            "task_name": task_name,
            "func_name": f.name,
            "weight": 1,
            "complexity": f.workload_complexity,
        })
    return extracted


def count_and_verify_tasks(ir_module, expected_count, expected_names):
    tasks = extract_tasks(ir_module)
    if len(tasks) != expected_count:
        return False, f"Expected {expected_count} tasks, got {len(tasks)}"
    names = [t["task_name"] for t in tasks]
    if sorted(names) != sorted(expected_names):
        return False, f"Expected task names {sorted(expected_names)}, got {sorted(names)}"
    return True, "Verified"
