import dis


def _classify_one(name):
    snippets = {
        "STORE": "def f():\n    x = 1\n",
        "list.append": "def f():\n    x = []\n    x.append(1)\n",
        "x+=1": "def f():\n    x = 0\n    x += 1\n",
        "d[k]+=1": "def f():\n    d = {0: 0}\n    d[0] += 1\n",
    }
    ns = {}
    exec(compile(snippets[name], "<classifier>", "exec"), ns)
    instructions = [
        ins
        for ins in dis.Bytecode(ns["f"])
        if ins.opname not in {"RESUME", "RETURN_VALUE", "RETURN_CONST"}
    ]
    return len(instructions) == 1


def classify_atomic_ops(ops):
    return {name: _classify_one(name) for name in ops}
