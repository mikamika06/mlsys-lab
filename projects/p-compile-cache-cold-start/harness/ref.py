import json
import os

class OracleEngine:
    def __init__(self):
        self.cache = {}
        self.compiled_count = 0
        self.version = "v1"

    def trace_ops(self, graph):
        return [op for op in graph if op not in self.cache]

    def compile_and_run(self, op):
        if op in self.cache:
            return self.cache[op], 0
        self.compiled_count += 1
        res = op * 2
        self.cache[op] = res
        return res, 1

    def export_cache(self, path):
        with open(path, "w") as f:
            json.dump(self.cache, f)

    def import_cache(self, path):
        if os.path.exists(path):
            with open(path, "r") as f:
                self.cache = json.load(f)

    def warmup(self, graphs):
        for g in graphs:
            for op in g:
                self.compile_and_run(op)

    def invalidate(self, new_version):
        if new_version != self.version:
            self.cache.clear()
            self.version = new_version
