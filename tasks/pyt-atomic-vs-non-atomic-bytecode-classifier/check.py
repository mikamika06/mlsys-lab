import dis


def _oracle_classify(ops):
    snippets = {
        "STORE": "def f():\n    x = 1\n",
        "list.append": "def f():\n    x = []\n    x.append(1)\n",
        "x+=1": "def f():\n    x = 0\n    x += 1\n",
        "d[k]+=1": "def f():\n    d = {0: 0}\n    d[0] += 1\n",
    }

    out = {}
    for name in ops:
        ns = {}
        exec(compile(snippets[name], "<oracle>", "exec"), ns)
        instructions = [
            ins
            for ins in dis.Bytecode(ns["f"])
            if ins.opname not in {"RESUME", "RETURN_VALUE", "RETURN_CONST"}
        ]
        out[name] = len(instructions) == 1
    return out


def grade(sol, fx) -> dict:
    cases = [
        ["STORE"],
        ["list.append"],
        ["x+=1"],
        ["d[k]+=1"],
        ["STORE", "list.append", "x+=1", "d[k]+=1"],
    ]

    ok = 1.0
    for ops in cases:
        try:
            got = sol.classify_atomic_ops(list(ops))
        except Exception:
            ok = 0.0
            break
        if got != _oracle_classify(ops):
            ok = 0.0
            break
    return {"exact_match": ok}
