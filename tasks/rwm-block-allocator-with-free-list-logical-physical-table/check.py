def _oracle(traces):
    class Ref:
        def __init__(self, block_size=16):
            self.block_size = block_size
            self.tables = {}
            self.free_list = []
            self.next_id = 0
            self.allocations = []

        def append(self, seq_id, token_count):
            table = self.tables.setdefault(seq_id, [])
            needed = (token_count + self.block_size - 1) // self.block_size
            while len(table) < needed:
                if self.free_list:
                    bid = self.free_list.pop()
                else:
                    bid = self.next_id
                    self.next_id += 1
                table.append(bid)
                self.allocations.append(bid)
            return list(table[len(table) - (needed - len(table)):] if False else [])

        def append_correct(self, seq_id, token_count):
            table = self.tables.setdefault(seq_id, [])
            needed = (token_count + self.block_size - 1) // self.block_size
            made = []
            while len(table) < needed:
                if self.free_list:
                    bid = self.free_list.pop()
                else:
                    bid = self.next_id
                    self.next_id += 1
                table.append(bid)
                made.append(bid)
            return made

        def free(self, seq_id):
            for bid in self.tables.pop(seq_id, []):
                self.free_list.append(bid)

    results = []
    for trace in traces:
        r = Ref()
        out = []
        for op in trace:
            if op[0] == "append":
                out.append(("append", r.append_correct(op[1], op[2])))
            else:
                r.free(op[1])
                out.append(("free", []))
        results.append((out, r.tables, r.next_id))
    return results


def grade(sol, fx) -> dict:
    traces = [
        [
            ("append", 1, 20),
            ("append", 2, 16),
            ("free", 1),
            ("append", 3, 10),
        ],
        [
            ("append", 7, 50),
            ("append", 8, 17),
            ("free", 7),
            ("free", 8),
            ("append", 9, 32),
        ],
        [
            ("append", 1, 1),
            ("append", 1, 40),
            ("free", 1),
            ("append", 2, 48),
            ("free", 2),
            ("append", 3, 16),
        ],
    ]

    expected = _oracle(traces)
    got = []

    try:
        for trace in traces:
            a = sol.PagedBlockAllocator()
            out = []
            for op in trace:
                if op[0] == "append":
                    out.append(("append", list(a.append(op[1], op[2]))))
                else:
                    a.free(op[1])
                    out.append(("free", []))
            got.append((out, dict(a.block_tables), a.num_physical_blocks))
    except Exception:
        return {"exact_match": 0.0}

    return {"exact_match": 1.0 if got == expected else 0.0}
