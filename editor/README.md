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
