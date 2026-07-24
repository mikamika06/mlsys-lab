import importlib
import os
import sys
import tempfile


def _make_module(directory, name):
    counter = os.path.join(directory, "count.txt")
    code = (
        "import os\n"
        "path = os.environ['ARENA_COUNT_FILE']\n"
        "try:\n"
        "    with open(path, 'r', encoding='utf-8') as f:\n"
        "        n = int(f.read())\n"
        "except FileNotFoundError:\n"
        "    n = 0\n"
        "with open(path, 'w', encoding='utf-8') as f:\n"
        "    f.write(str(n + 1))\n"
        "EXECUTIONS = n + 1\n"
    )
    with open(os.path.join(directory, name + ".py"), "w", encoding="utf-8") as f:
        f.write(code)
    return counter


def _oracle(module_name, module_dir):
    old_path = list(sys.path)
    old_env = dict(os.environ)
    try:
        count_file = os.environ["ARENA_COUNT_FILE"]
        sys.modules.pop(module_name, None)
        sys.path.insert(0, module_dir)
        first = importlib.import_module(module_name)
        second = importlib.import_module(module_name)
        if first is not second:
            raise RuntimeError("imports did not return the cached module object")
        return int(second.EXECUTIONS)
    finally:
        sys.path[:] = old_path
        os.environ.clear()
        os.environ.update(old_env)
        sys.modules.pop(module_name, None)


def grade(sol, fx) -> dict:
    ok = 1.0
    try:
        with tempfile.TemporaryDirectory() as d:
            name = "arena_counter_module"
            count_file = _make_module(d, name)
            os.environ["ARENA_COUNT_FILE"] = count_file
            expected = _oracle(name, d)
            with open(count_file, "w", encoding="utf-8") as f:
                f.write("0")
            got = sol.cached_import_count(name, d)
            ok = 1.0 if got == expected else 0.0
    except Exception:
        ok = 0.0
    return {"exact_match": ok}
