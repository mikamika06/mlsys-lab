import random

class OracleAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks - 1, -1, -1))
        self.ref_counts = {i: 0 for i in range(num_blocks)}
        self.seq_lengths = {}
        self.block_tables = {}
        self.next_seq_id = 1

    def free_count(self) -> int:
        return len(self.free_blocks)

    def allocate_sequence(self) -> int:
        sid = self.next_seq_id
        self.next_seq_id += 1
        self.seq_lengths[sid] = 0
        self.block_tables[sid] = []
        return sid

    def get_block_table(self, seq_id: int) -> list[int]:
        return self.block_tables[seq_id].copy()

    def get_block_refcount(self, block_id: int) -> int:
        return self.ref_counts[block_id]

    def append_tokens(self, seq_id: int, num_tokens: int):
        for _ in range(num_tokens):
            curr_len = self.seq_lengths[seq_id]
            block_idx = curr_len // self.block_size
            if block_idx == len(self.block_tables[seq_id]):
                b = self.free_blocks.pop()
                self.ref_counts[b] = 1
                self.block_tables[seq_id].append(b)
            else:
                b = self.block_tables[seq_id][block_idx]
                if self.ref_counts[b] > 1:
                    b_new = self.free_blocks.pop()
                    self.ref_counts[b] -= 1
                    self.ref_counts[b_new] = 1
                    self.block_tables[seq_id][block_idx] = b_new
            self.seq_lengths[seq_id] += 1

    def fork_sequence(self, parent_id: int) -> int:
        sid = self.next_seq_id
        self.next_seq_id += 1
        self.seq_lengths[sid] = self.seq_lengths[parent_id]
        self.block_tables[sid] = self.block_tables[parent_id].copy()
        for b in self.block_tables[sid]:
            self.ref_counts[b] += 1
        return sid

    def free_sequence(self, seq_id: int):
        for b in self.block_tables[seq_id]:
            self.ref_counts[b] -= 1
            if self.ref_counts[b] == 0:
                self.free_blocks.append(b)
        del self.seq_lengths[seq_id]
        del self.block_tables[seq_id]

def get_random_trace(steps: int) -> list:
    random.seed(42)
    trace = []
    active = []
    seq_names = 0
    for _ in range(steps):
        op = random.random()
        if not active or op < 0.2:
            name = f"s{seq_names}"
            seq_names += 1
            trace.append(("alloc", name))
            active.append(name)
        elif op < 0.6:
            name = random.choice(active)
            trace.append(("append", name, random.randint(1, 5)))
        elif op < 0.8:
            src = random.choice(active)
            dst = f"s{seq_names}"
            seq_names += 1
            trace.append(("fork", src, dst))
            active.append(dst)
        else:
            name = random.choice(active)
            active.remove(name)
            trace.append(("free", name))
    for name in active:
        trace.append(("free", name))
    return trace

def get_beam_search_trace() -> list:
    trace = []
    trace.append(("alloc", "b0"))
    trace.append(("append", "b0", 100))
    for i in range(1, 9):
        trace.append(("fork", "b0", f"b{i}"))
    trace.append(("free", "b0"))
    active = [f"b{i}" for i in range(1, 9)]
    seq_names = 9
    for _ in range(20):
        for name in active:
            trace.append(("append", name, 1))
        trace.append(("free", active[0]))
        trace.append(("free", active[1]))
        new_1 = f"b{seq_names}"
        new_2 = f"b{seq_names+1}"
        seq_names += 2
        trace.append(("fork", active[2], new_1))
        trace.append(("fork", active[3], new_2))
        active = active[2:] + [new_1, new_2]
    for name in active:
        trace.append(("free", name))
    return trace

def run_trace(alloc_cls, num_blocks, block_size, trace):
    alloc = alloc_cls(num_blocks, block_size)
    mapping = {}
    history = []
    for instr in trace:
        op = instr[0]
        if op == "alloc":
            mapping[instr[1]] = alloc.allocate_sequence()
        elif op == "append":
            alloc.append_tokens(mapping[instr[1]], instr[2])
        elif op == "fork":
            mapping[instr[2]] = alloc.fork_sequence(mapping[instr[1]])
        elif op == "free":
            alloc.free_sequence(mapping[instr[1]])
            del mapping[instr[1]]
        history.append(alloc.free_count())
    return history
