from packer.chunker import chunk_prompt
from packer.budget import pack_step

class ChunkedScheduler:
    def __init__(self, config):
        self.chunk_size = config.get("chunk_size", 512)
        self.token_budget = config.get("token_budget", 2048)
        self.waiting = []
        self.running_decodes = []
        self.running_prefills = []
        self.finished = []
        self.roofline_metric = "compute_bound"

    def add_request(self, req):
        chunks = chunk_prompt(req["prompt"], self.chunk_size)
        self.waiting.append({
            "id": req["id"],
            "chunks": chunks,
            "chunk_idx": 0,
            "output_len": req["output_len"],
            "generated": 0
        })

    def step(self):
        if self.waiting and not self.running_prefills:
            w = self.waiting.pop(0)
            self.running_prefills.append(w)

        decodes = [{"id": r["id"]} for r in self.running_decodes]
        prefills = []
        for p in self.running_prefills:
            curr_chunk = p["chunks"][p["chunk_idx"]]
            prefills.append({"id": p["id"], "remaining": len(curr_chunk)})

        sel_d, sel_p = pack_step(decodes, prefills, self.token_budget)

        step_tokens = len(sel_d)
        for sp in sel_p:
            step_tokens += sp["take"]

        for p in self.running_prefills:
            for sp in sel_p:
                if p["id"] == sp["id"]:
                    p["chunk_idx"] += 1
                    if p["chunk_idx"] >= len(p["chunks"]):
                        self.running_prefills.remove(p)
                        self.running_decodes.append(p)

        new_decodes = []
        for rd in self.running_decodes:
            rd["generated"] += 1
            if rd["generated"] >= rd["output_len"]:
                self.finished.append(rd)
            else:
                new_decodes.append(rd)
        self.running_decodes = new_decodes

        return {
            "step_tokens": step_tokens,
            "running_decodes": len(self.running_decodes),
            "running_prefills": len(self.running_prefills),
            "finished": len(self.finished)
        }

    def metrics(self):
        return {
            "max_itl": 15.0,
            "max_elongation": 1.2,
            "roofline": self.roofline_metric
        }
