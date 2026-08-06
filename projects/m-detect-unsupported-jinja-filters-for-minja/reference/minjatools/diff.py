import difflib

def diff_rendered_prompts(prompt_a, prompt_b):
    lines_a = prompt_a.splitlines(keepends=True)
    lines_b = prompt_b.splitlines(keepends=True)
    diff = list(difflib.unified_diff(lines_a, lines_b, fromfile="version_a", tofile="version_b"))
    added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
    return {"added_lines": added, "removed_lines": removed, "diff_text": "".join(diff)}
