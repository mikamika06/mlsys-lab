from guardeval.evaluator import evaluate_graph_guards


class Engine:
    def __init__(self, compile_fn):
        self.compile_fn = compile_fn
        self.compiled_graphs = []
        self.recompile_count = 0
        self.attribution_log = []

    def process_stream(self, input_stream):
        for idx, meta in enumerate(input_stream):
            matched = False
            first_failures = []
            for g_idx, graph in enumerate(self.compiled_graphs):
                ok, reason = evaluate_graph_guards(graph["guards"], meta)
                if ok:
                    matched = True
                    break
                else:
                    first_failures.append((g_idx, reason))

            if not matched:
                self.recompile_count += 1
                primary_reason = first_failures[0][1] if first_failures else "no_existing_graph"
                self.attribution_log.append({
                    "step": idx,
                    "reason": primary_reason,
                    "meta": meta
                })
                new_graph = self.compile_fn(meta)
                self.compiled_graphs.append(new_graph)

        return {
            "recompile_count": self.recompile_count,
            "attributions": self.attribution_log,
            "num_graphs": len(self.compiled_graphs)
        }
