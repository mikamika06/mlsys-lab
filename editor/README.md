# mlsys-lab — VS Code extension

The workspace UI: a roadmap of the task bank, the statement, an editor and the
grader's verdict, side by side.

## Run it

```bash
npx @vscode/vsce package
code --install-extension mlsys-lab-0.1.0.vsix
```

Open the **repository root** (the folder containing `src/mlsys/` and `tasks/`)
and run `mlsys-lab: Open Workspace`.

## Run vs Grade

Two buttons in the toolbar. **Grade** (`⌘↵`) measures the solution against the
task's gates and is the only thing that marks a task solved. **Run** (`⇧⌘↵`)
executes the file as a plain script and streams whatever it prints — the way to
see a `print`, a shape, or a traceback while working.

Run executes python tasks; a `.cpp` or `.cu` needs a compile step and the button
says so rather than pretending. While a process is alive the button becomes
**Stop**, and it is killed anyway after `mlsys.runTimeoutSeconds` (30 by default)
or 200 KB of output, so a runaway loop cannot wedge the editor.

## How grading is dispatched

The extension shells out to the engine; the language comes from the task's
`meta.json`:

| `meta.native` | file you edit | command |
|---|---|---|
| *(absent)* | `solve.py` | `python -m mlsys grade <id> --file solve.py --json` |
| `"cpp"` | `solve.cpp` | `python -m mlsys.runners.cpp tasks/<id> solve.cpp` |
| `"cuda"` | `solve.cu` | `python -m mlsys.runners.cuda tasks/<id> solve.cu` |

`PYTHONPATH` is set to the repository's `src/`, so a checkout works without
installing the package.

## Settings

- `mlsys.pythonPath` — interpreter used to run the grader (default `python3`)
- `mlsys.workDir` — where your solutions go, one folder per task (default `~/mlsys-lab`)
- `mlsys.runTimeoutSeconds` — how long **Run** lets your code execute (default `30`)
