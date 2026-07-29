from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Request:
    rid: str
    arrival: int
    prompt: tuple[int, ...]
    output_len: int

    @property
    def prompt_len(self) -> int:
        return len(self.prompt)


@dataclass
class ServerConfig:
    block_size: int = 16
    num_blocks: int = 512
    max_batch_tokens: int = 2048
    max_seqs: int = 16
    chunked_prefill: bool = True
    prefix_cache: bool = False
    preemption: str = "recompute"
    swap_blocks: int = 0
    max_preemptions: int = 8
    prefill_cost: int = 1
    decode_cost: int = 8
    step_overhead: int = 2


class OutOfBlocks(Exception):
    pass


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free = list(range(num_blocks))
        self.ref = [0] * num_blocks
        self.hash_to_block: dict[str, int] = {}
        self.block_to_hash: dict[int, str] = {}

    @property
    def used(self) -> int:
        return self.num_blocks - len(self.free)

    def utilisation(self) -> float:
        return self.used / self.num_blocks if self.num_blocks else 0.0

    def allocate(self) -> int:
        if not self.free:
            raise OutOfBlocks()
        b = self.free.pop(0)
        self.ref[b] = 1
        return b

    def share(self, block: int) -> int:
        self.ref[block] += 1
        return block

    def release(self, block: int) -> None:
        if self.ref[block] == 0:
            return
        self.ref[block] -= 1
        if self.ref[block] == 0:
            h = self.block_to_hash.pop(block, None)
            if h is not None and self.hash_to_block.get(h) == block:
                del self.hash_to_block[h]
            self.free.append(block)
            self.free.sort()

    def register(self, block: int, h: str) -> None:
        self.hash_to_block[h] = block
        self.block_to_hash[block] = h

    def lookup(self, h: str) -> int | None:
        return self.hash_to_block.get(h)


def block_hashes(tokens: tuple[int, ...], block_size: int, tenant: str = "") -> list[str]:
    out: list[str] = []
    parent = tenant
    for i in range(0, len(tokens) - len(tokens) % block_size, block_size):
        chunk = tokens[i:i + block_size]
        parent = hashlib.sha256(
            (parent + "|" + ",".join(str(t) for t in chunk)).encode()
        ).hexdigest()[:16]
        out.append(parent)
    return out


@dataclass
class _Seq:
    req: Request
    blocks: list[int] = field(default_factory=list)
    cached_tokens: int = 0
    prefilled: int = 0
    decoded: int = 0
    admitted_at: int = -1
    first_token_at: int = -1
    finished_at: int = -1
    preempted: int = 0

    @property
    def total_tokens(self) -> int:
        return self.req.prompt_len + self.decoded

    @property
    def done(self) -> bool:
        return self.decoded >= self.req.output_len


class Server:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.alloc = BlockAllocator(cfg.num_blocks, cfg.block_size)
        self.time = 0
        self.waiting: list[_Seq] = []
        self.running: list[_Seq] = []
        self.finished: list[_Seq] = []
        self.swapped: list[_Seq] = []
        self.rejected: list[_Seq] = []
        self.trace: list[dict] = []
        self.stats = {
            "steps": 0, "preemptions": 0, "swaps": 0, "recomputes": 0,
            "cached_tokens": 0, "prefill_tokens": 0, "decode_tokens": 0,
            "rejected": 0,
        }

    def _blocks_for(self, n_tokens: int) -> int:
        bs = self.cfg.block_size
        return (n_tokens + bs - 1) // bs

    def _grow(self, seq: _Seq, upto_tokens: int) -> bool:
        need = self._blocks_for(upto_tokens) - len(seq.blocks)
        if need <= 0:
            return True
        if need > len(self.alloc.free):
            return False
        for _ in range(need):
            seq.blocks.append(self.alloc.allocate())
        return True

    def _free(self, seq: _Seq) -> None:
        for b in seq.blocks:
            self.alloc.release(b)
        seq.blocks = []

    def _try_prefix_cache(self, seq: _Seq) -> None:
        if not self.cfg.prefix_cache:
            return
        hashes = block_hashes(seq.req.prompt, self.cfg.block_size)
        hit = 0
        for h in hashes:
            b = self.alloc.lookup(h)
            if b is None:
                break
            seq.blocks.append(self.alloc.share(b))
            hit += 1
        seq.cached_tokens = hit * self.cfg.block_size
        seq.prefilled = seq.cached_tokens

    def _register_prefix(self, seq: _Seq) -> None:
        if not self.cfg.prefix_cache:
            return
        hashes = block_hashes(seq.req.prompt, self.cfg.block_size)
        for i, h in enumerate(hashes):
            if i < len(seq.blocks) and self.alloc.lookup(h) is None:
                self.alloc.register(seq.blocks[i], h)

    def _reject(self, seq: _Seq) -> None:
        self.stats["rejected"] += 1
        self._free(seq)
        if seq in self.running:
            self.running.remove(seq)
        self.rejected.append(seq)

    def _preempt(self, victim: _Seq) -> None:
        if victim not in self.running:
            return
        self.running.remove(victim)
        self.stats["preemptions"] += 1
        victim.preempted += 1
        if victim.preempted > self.cfg.max_preemptions:
            self._reject(victim)
            return
        if self.cfg.preemption == "swap" and self.cfg.swap_blocks >= len(victim.blocks):
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

    def _admit(self) -> None:
        cfg = self.cfg
        pool = self.swapped + self.waiting
        for seq in list(pool):
            if len(self.running) >= cfg.max_seqs:
                return
            if seq.req.arrival > self.time:
                continue
            if seq in self.swapped:
                if not self._grow(seq, seq.total_tokens):
                    continue
                self.swapped.remove(seq)
                self.running.append(seq)
                continue
            self._try_prefix_cache(seq)
            if not self._grow(seq, seq.req.prompt_len):
                self._free(seq)
                seq.prefilled = 0
                seq.cached_tokens = 0
                continue
            if seq.admitted_at < 0:
                seq.admitted_at = self.time
            self.stats["cached_tokens"] += seq.cached_tokens
            self.waiting.remove(seq)
            self.running.append(seq)

    def step(self) -> dict:
        cfg = self.cfg
        self._admit()
        budget = cfg.max_batch_tokens
        prefill_tokens = 0
        decode_tokens = 0
        did = []

        for seq in list(self.running):
            if seq.prefilled >= seq.req.prompt_len:
                continue
            remaining = seq.req.prompt_len - seq.prefilled
            take = min(remaining, budget) if cfg.chunked_prefill else remaining
            if take <= 0 or take > budget:
                continue
            seq.prefilled += take
            budget -= take
            prefill_tokens += take
            did.append(seq.req.rid)
            if seq.prefilled >= seq.req.prompt_len:
                self._register_prefix(seq)

        for seq in list(self.running):
            if seq not in self.running or seq.prefilled < seq.req.prompt_len:
                continue
            if budget < 1:
                break
            if not self._grow(seq, seq.total_tokens + 1):
                if len(self.running) == 1:
                    self._reject(seq)
                    continue
                victim = self._victim()
                self._preempt(victim if victim is not seq else self._victim_other(seq))
                continue
            seq.decoded += 1
            budget -= 1
            decode_tokens += 1
            did.append(seq.req.rid)
            if seq.first_token_at < 0:
                seq.first_token_at = self.time
            if seq.done:
                seq.finished_at = self.time
                self._free(seq)
                self.running.remove(seq)
                self.finished.append(seq)

        cost = (cfg.step_overhead
                + prefill_tokens * cfg.prefill_cost
                + decode_tokens * cfg.decode_cost)
        rec = {
            "t": self.time, "step": self.stats["steps"],
            "running": len(self.running), "waiting": len(self.waiting),
            "prefill_tokens": prefill_tokens, "decode_tokens": decode_tokens,
            "blocks_used": self.alloc.used, "cost": cost,
            "ids": tuple(did),
        }
        self.trace.append(rec)
        self.stats["steps"] += 1
        self.stats["prefill_tokens"] += prefill_tokens
        self.stats["decode_tokens"] += decode_tokens
        self.time += cost if cost > cfg.step_overhead else 1
        return rec

    def _victim(self) -> _Seq:
        return max(self.running, key=lambda s: (s.total_tokens, s.req.rid))

    def _victim_other(self, keep: _Seq) -> _Seq:
        others = [s for s in self.running if s is not keep]
        return max(others, key=lambda s: (s.total_tokens, s.req.rid)) if others else keep

    def run(self, requests: list[Request], max_steps: int = 100000) -> dict:
        self.waiting = [_Seq(r) for r in sorted(requests, key=lambda r: (r.arrival, r.rid))]
        for _ in range(max_steps):
            if not self.waiting and not self.running and not self.swapped:
                break
            before = (len(self.finished), self.time)
            self.step()
            if (not self.running and not self.swapped and self.waiting
                    and all(s.req.arrival > self.time for s in self.waiting)):
                self.time = min(s.req.arrival for s in self.waiting)
            if before == (len(self.finished), self.time) and not self.running:
                self.time += 1
        return self.metrics()

    def metrics(self) -> dict:
        fin = self.finished
        if not fin:
            return {"finished": 0, **{k: float("inf") for k in
                                      ("ttft_p50", "ttft_p95", "latency_p50", "latency_p95",
                                       "latency_p99", "tpot_mean")},
                    "throughput": 0.0, "block_utilisation": self.alloc.utilisation(),
                    **self.stats}

        def pct(xs, q):
            xs = sorted(xs)
            if not xs:
                return float("inf")
            i = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
            return float(xs[i])

        ttft = [s.first_token_at - s.req.arrival for s in fin]
        lat = [s.finished_at - s.req.arrival for s in fin]
        tpot = [(s.finished_at - s.first_token_at) / max(1, s.decoded - 1) for s in fin]
        span = max(1, max(s.finished_at for s in fin))
        out_tokens = sum(s.decoded for s in fin)
        return {
            "finished": len(fin),
            "ttft_p50": pct(ttft, 0.50), "ttft_p95": pct(ttft, 0.95),
            "latency_p50": pct(lat, 0.50), "latency_p95": pct(lat, 0.95),
            "latency_p99": pct(lat, 0.99),
            "tpot_mean": sum(tpot) / len(tpot),
            "throughput": out_tokens / span,
            "block_utilisation": self.alloc.utilisation(),
            "cache_hit_rate": (self.stats["cached_tokens"] /
                               max(1, self.stats["cached_tokens"] + self.stats["prefill_tokens"])),
            **self.stats,
        }


def simulate(requests, config=None, **kw) -> dict:
    cfg = config or ServerConfig(**kw)
    return Server(cfg).run(list(requests))


def make_requests(spec) -> list[Request]:
    out = []
    for i, s in enumerate(spec):
        prompt = s.get("prompt")
        if prompt is None:
            n = s["prompt_len"]
            base = s.get("prefix", ())
            prompt = tuple(base) + tuple(range(1000 + i * 100, 1000 + i * 100 + n - len(base)))
        out.append(Request(rid=s.get("rid", f"r{i}"), arrival=s.get("arrival", 0),
                           prompt=tuple(prompt), output_len=s["output_len"]))
    return out
