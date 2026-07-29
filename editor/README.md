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

One button, three tracks. `python -m mlsys.runners.run` decides how each is
actually run and prints the command it used:

| track | what Run does |
|---|---|
| python | executes `solve.py` as a script |
| `cpp` | `clang++ -O2 -std=c++20` against the task's `main.cpp` driver, then the binary |
| `cuda` | parses `solve.cu`, lists the kernels, executes it on the software GPU |

A `.cpp` that defines its own `main()` is compiled alone; one that does not is
linked against the task's driver, so a contract implementation still runs. CUDA
shows the counters the grader measures and no verdict — gates are grading.

While a process is alive the button becomes **Stop**, which kills the whole
process group rather than orphaning the child. It is killed anyway after
`mlsys.runTimeoutSeconds` (30 by default) or 200 KB of output, so a runaway loop
cannot wedge the editor. Run needs `mlsys-lab >= 0.1.2`.

## Typing

The editor is a textarea, so the courtesies it does not come with are written by
hand. Brackets and quotes close themselves — typing the closer that is already
there steps over it, backspace between a pair removes both halves, and a
selection is wrapped rather than replaced. A quote following a word character is
left alone, so `don't` and a C++ char literal type normally.

`Enter` keeps the indentation of the line you were on, and adds a level after a
`{`, `(`, `[` or, in python, a `:`. With the caret between a pair the closer gets
its own line. `Tab` indents, `⇧Tab` dedents, and backspace inside indentation
removes a whole level.

Every edit goes through `execCommand('insertText')` — deprecated, and the only
way to modify a textarea without clearing the native undo stack.

## Projects

Part-2 projects are multi-file, so the panel does not try to be their editor: it
shows the ticket, the file list and the milestones, and the code is edited in VS
Code itself. **Start** copies the skeleton into `mlsys.workDir` and opens the brief;
each milestone grades on its own and remembers what it cleared. Progress is stored
per project and declared for Settings Sync when the first milestone is cleared —
the key does not exist before that, so declaring it once at activation would have
dropped it.

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
