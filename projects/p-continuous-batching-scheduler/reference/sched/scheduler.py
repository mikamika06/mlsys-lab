import hashlib

from .allocator import Allocator
from . import policy

DEFAULTS = {
    "block_size": 16, "num_blocks": 512, "max_batch_tokens": 2048, "max_seqs": 16,
    "chunked_prefill": True, "prefix_cache": False, "preemption": "recompute",
    "swap_blocks": 0, "max_preemptions": 8, "prefill_cost": 1, "decode_cost": 8,
    "step_overhead": 2,
}


def block_keys(tokens, block_size, tenant=""):
    out = []
    parent = tenant
    for i in range(0, len(tokens) - len(tokens) % block_size, block_size):
        chunk = tokens[i:i + block_size]
        parent = hashlib.sha256(
            (parent + "|" + ",".join(str(t) for t in chunk)).encode()).hexdigest()[:16]
        out.append(parent)
    return out


class Seq:
    def __init__(self, req):
        self.rid = req["rid"]
        self.arrival = req["arrival"]
        self.prompt = tuple(req["prompt"])
        self.output_len = req["output_len"]
        self.blocks = []
        self.cached_tokens = 0
        self.prefilled = 0
        self.decoded = 0
        self.admitted_at = -1
        self.first_token_at = -1
        self.finished_at = -1
        self.preempted = 0

    @property
    def prompt_len(self):
        return len(self.prompt)

    @property
    def total_tokens(self):
        return self.prompt_len + self.decoded

    @property
    def done(self):
        return self.decoded >= self.output_len


class Scheduler:
    def __init__(self, config: dict):
        self.cfg = dict(DEFAULTS)
        self.cfg.update(config or {})
        self.alloc = Allocator(self.cfg["num_blocks"], self.cfg["block_size"])
        self.time = 0
        self.waiting = []
        self.running = []
        self.finished = []
        self.swapped = []
        self.rejected = []
        self.trace = []
        self.stats = {"steps": 0, "preemptions": 0, "swaps": 0, "recomputes": 0,
                      "cached_tokens": 0, "prefill_tokens": 0, "decode_tokens": 0,
                      "rejected": 0}

    def add(self, requests) -> None:
        self.waiting.extend(Seq(r) for r in requests)
        self.waiting.sort(key=lambda s: (s.arrival, s.rid))

    def _blocks_for(self, n):
        bs = self.cfg["block_size"]
        return (n + bs - 1) // bs

    def _grow(self, seq, upto):
        need = self._blocks_for(upto) - len(seq.blocks)
        if need <= 0:
            return True
        if need > self.alloc.free_count():
            return False
        for _ in range(need):
            seq.blocks.append(self.alloc.allocate())
        return True

    def _free(self, seq):
        for b in seq.blocks:
            self.alloc.release(b)
        seq.blocks = []

    def _reject(self, seq):
        self.stats["rejected"] += 1
        self._free(seq)
        if seq in self.running:
            self.running.remove(seq)
        self.rejected.append(seq)

    def _preempt(self, victim):
        if victim not in self.running:
            return
        self.running.remove(victim)
        self.stats["preemptions"] += 1
        victim.preempted += 1
        if victim.preempted > self.cfg["max_preemptions"]:
            self._reject(victim)
            return
        if self.cfg["preemption"] == "swap" and self.cfg["swap_blocks"] >= len(victim.blocks):
            self.stats["swaps"] += 1
            self._free(victim)
            self.swapped.append(victim)
        else:
            self.stats["recomputes"] += 1
            self._free(victim)
            victim.prefilled = 0
            victim.cached_tokens = 0
            victim.decoded = 0
            self.waiting.insert(0, victim)

    def _try_prefix(self, seq):
        if not self.cfg["prefix_cache"]:
            return
        hit = 0
        for key in block_keys(seq.prompt, self.cfg["block_size"]):
            b = self.alloc.lookup(key)
            if b is None:
                break
            seq.blocks.append(self.alloc.share(b))
            hit += 1
        seq.cached_tokens = hit * self.cfg["block_size"]
        seq.prefilled = seq.cached_tokens

    def _register_prefix(self, seq):
        if not self.cfg["prefix_cache"]:
            return
        for i, key in enumerate(block_keys(seq.prompt, self.cfg["block_size"])):
            if i < len(seq.blocks) and self.alloc.lookup(key) is None:
                self.alloc.register(seq.blocks[i], key)

    def _admit(self):
        for seq in list(self.swapped) + list(self.waiting):
            if seq.arrival > self.time:
                continue
            if seq in self.swapped:
                state = {"running": len(self.running), "max_seqs": self.cfg["max_seqs"],
                         "free_blocks": self.alloc.free_count(),
                         "blocks_needed": self._blocks_for(seq.total_tokens) - len(seq.blocks)}
                if not policy.should_admit(state):
                    if state["running"] >= state["max_seqs"]:
                        return
                    continue
                if not self._grow(seq, seq.total_tokens):
                    continue
                self.swapped.remove(seq)
                self.running.append(seq)
                continue
            self._try_prefix(seq)
            state = {"running": len(self.running), "max_seqs": self.cfg["max_seqs"],
                     "free_blocks": self.alloc.free_count(),
                     "blocks_needed": self._blocks_for(seq.prompt_len) - len(seq.blocks)}
            if not policy.should_admit(state):
                self._free(seq)
                seq.prefilled = 0
                seq.cached_tokens = 0
                if state["running"] >= state["max_seqs"]:
                    return
                continue
            if not self._grow(seq, seq.prompt_len):
                self._free(seq)
                seq.prefilled = 0
                seq.cached_tokens = 0
                continue
            if seq.admitted_at < 0:
                seq.admitted_at = self.time
            self.stats["cached_tokens"] += seq.cached_tokens
            self.waiting.remove(seq)
            self.running.append(seq)

    def _victim_other(self, keep):
        others = [s for s in self.running if s is not keep]
        return policy.victim(others) if others else keep

    def step(self) -> dict:
        cfg = self.cfg
        self._admit()
        budget = cfg["max_batch_tokens"]
        pre = dec = 0
        did = []

        for seq in list(self.running):
            if seq.prefilled >= seq.prompt_len:
                continue
            remaining = seq.prompt_len - seq.prefilled
            take = min(remaining, budget) if cfg["chunked_prefill"] else remaining
            if take <= 0 or take > budget:
                continue
            seq.prefilled += take
            budget -= take
            pre += take
            did.append(seq.rid)
            if seq.prefilled >= seq.prompt_len:
                self._register_prefix(seq)

        for seq in list(self.running):
            if seq not in self.running or seq.prefilled < seq.prompt_len:
                continue
            if budget < 1:
                break
            if not self._grow(seq, seq.total_tokens + 1):
                if len(self.running) == 1:
                    self._reject(seq)
                    continue
                v = policy.victim(self.running)
                self._preempt(v if v is not seq else self._victim_other(seq))
                continue
            seq.decoded += 1
            budget -= 1
            dec += 1
            did.append(seq.rid)
            if seq.first_token_at < 0:
                seq.first_token_at = self.time
            if seq.done:
                seq.finished_at = self.time
                self._free(seq)
                self.running.remove(seq)
                self.finished.append(seq)

        cost = cfg["step_overhead"] + pre * cfg["prefill_cost"] + dec * cfg["decode_cost"]
        rec = {"t": self.time, "step": self.stats["steps"], "running": len(self.running),
               "waiting": len(self.waiting), "prefill_tokens": pre, "decode_tokens": dec,
               "blocks_used": self.alloc.num_blocks - self.alloc.free_count(),
               "cost": cost, "ids": tuple(did)}
        self.trace.append(rec)
        self.stats["steps"] += 1
        self.stats["prefill_tokens"] += pre
        self.stats["decode_tokens"] += dec
        self.time += cost if cost > cfg["step_overhead"] else 1
        return rec

    def run(self, max_steps: int = 100000) -> dict:
        for _ in range(max_steps):
            if not self.waiting and not self.running and not self.swapped:
                break
            before = (len(self.finished), self.time)
            self.step()
            if (not self.running and not self.swapped and self.waiting
                    and all(s.arrival > self.time for s in self.waiting)):
                self.time = min(s.arrival for s in self.waiting)
            if before == (len(self.finished), self.time) and not self.running:
                self.time += 1
        return self.metrics()

    def metrics(self) -> dict:
        fin = self.finished
        base = dict(self.stats)
        base["block_utilisation"] = ((self.alloc.num_blocks - self.alloc.free_count())
                                     / max(1, self.alloc.num_blocks))
        base["cache_hit_rate"] = (self.stats["cached_tokens"] /
                                  max(1, self.stats["cached_tokens"] + self.stats["prefill_tokens"]))
        if not fin:
            base.update(finished=0, ttft_p50=float("inf"), ttft_p95=float("inf"),
                        latency_p50=float("inf"), latency_p95=float("inf"),
                        latency_p99=float("inf"), tpot_mean=float("inf"), throughput=0.0)
            return base

        def pct(xs, q):
            xs = sorted(xs)
            i = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
            return float(xs[i])

        ttft = [s.first_token_at - s.arrival for s in fin]
        lat = [s.finished_at - s.arrival for s in fin]
        tpot = [(s.finished_at - s.first_token_at) / max(1, s.decoded - 1) for s in fin]
        span = max(1, max(s.finished_at for s in fin))
        base.update(finished=len(fin), ttft_p50=pct(ttft, 0.5), ttft_p95=pct(ttft, 0.95),
                    latency_p50=pct(lat, 0.5), latency_p95=pct(lat, 0.95),
                    latency_p99=pct(lat, 0.99), tpot_mean=sum(tpot) / len(tpot),
                    throughput=sum(s.decoded for s in fin) / span)
        return base
