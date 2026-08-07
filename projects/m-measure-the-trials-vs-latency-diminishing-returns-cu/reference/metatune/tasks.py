def extract_tasks(module_spec):
    tasks = []
    for name in module_spec:
        tasks.append({"task_name": f"f_{name}", "weight": 1.0})
    return tasks
