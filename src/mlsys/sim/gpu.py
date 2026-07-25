"""A software GPU — realistic SIMT simulator. Runs a kernel over a grid of
threads grouped into 32-lane warps, over a simulated global-memory BLOCK, and
reports REAL GPU behaviour: memory coalescing (128-byte transactions),
shared-memory bank conflicts, warp divergence, and a cycle estimate.

Self-contained, deterministic, no hardware. The learner writes a kernel; the
simulator executes it exactly like a GPU would and shows why coalesced access is
fast and strided access is slow — the real lesson, measured, identical everywhere.
"""
import inspect

import numpy as np

WARP = 32
SEGMENT = 128          # global-memory transaction size in bytes (like a real GPU)
BANKS = 32             # shared-memory banks (4-byte words)
CYC_MEM = 200          # global-memory transaction latency (cycles), amortized
CYC_SMEM = 20          # shared-memory access (cycles) per serialized wave
CYC_ALU = 1            # arithmetic op


class Idx(int):
    """A CUDA index that is simultaneously a plain int and a dim3.

    `t.threadIdx` has always been an int in this simulator, and 16 existing tasks
    plus the CUDA-C frontend rely on that. Real CUDA code writes `threadIdx.x`.
    Subclassing int gives both: arithmetic keeps working unchanged, while `.x`,
    `.y` and `.z` expose the 2D/3D view. A 1D launch reports y=z=0, exactly like
    the hardware does.
    """
    def __new__(cls, x, y=0, z=0):
        o = int.__new__(cls, x)
        o._y, o._z = int(y), int(z)
        return o

    @property
    def x(self):
        return int(self)

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    def __repr__(self):
        return f"Idx({int(self)}, {self._y}, {self._z})"


class ShflRequest:
    """A pending warp-shuffle, yielded by a generator kernel.

    Real `__shfl_up_sync` is warp-synchronous: every lane hands its value to the
    lane `delta` above it in the same instruction. That is only meaningful if the
    lanes are at the same program point, so it is implemented exactly like the
    `__syncthreads()` barrier — the kernel yields the request, `GPU.launch`
    collects every lane's value, then resumes each generator with its own result.
    """
    __slots__ = ("kind", "value", "delta")

    def __init__(self, kind, value, delta):
        self.kind, self.value, self.delta = kind, value, delta


class Thread:
    """The per-thread view the kernel sees — like CUDA's threadIdx/blockIdx."""
    def __init__(self, sim, tid, bid, bdim, gdim, tid_y=0, bid_y=0, bdim_y=1, gdim_y=1):
        self.threadIdx = Idx(tid, tid_y)
        self.blockIdx = Idx(bid, bid_y)
        self.blockDim = Idx(bdim, bdim_y)
        self.gridDim = Idx(gdim, gdim_y)
        # linear id: row-major over (y, x), matching how a 2D launch is flattened
        self.gid = (bid_y * gdim + bid) * (bdim * bdim_y) + tid_y * bdim + tid
        self.lane = (tid_y * bdim + tid) % WARP
        self._sim = sim
        self._g = []       # ordered list of global byte-addresses this thread touched
        self._s = []       # ordered list of shared word-indices
        self._alu = 0
        self._salloc_i = 0

    def gload(self, idx, itembytes=4):
        self._g.append(int(idx) * itembytes)
        return self._sim.gmem[int(idx)]

    def gstore(self, idx, val, itembytes=4):
        self._g.append(int(idx) * itembytes)
        self._sim._wrote(int(idx), self.gid, atomic=False)
        self._sim.gmem[int(idx)] = val

    # --- wide access. A float4 / half2 load is ONE instruction covering several
    # consecutive elements, so the warp issues one address for the whole vector
    # instead of one per element — which is the entire point of vectorising.
    def gload_wide(self, idx, n, itembytes=4):
        base = int(idx)
        self._g.append(base * itembytes)
        return [float(self._sim.gmem[base + k]) for k in range(n)]

    def gstore_wide(self, idx, vals, itembytes=4):
        base = int(idx)
        self._g.append(base * itembytes)
        for k, v in enumerate(vals):
            self._sim._wrote(base + k, self.gid, atomic=False)
            self._sim.gmem[base + k] = v

    # --- atomics. Threads here are executed one at a time, so an ordinary
    # read-modify-write would *appear* to work; the hazard is instead recorded
    # (see GPU._wrote / the `races` metric) and an atomic is what clears it.
    def atomic(self, op, idx, val, shared=False):
        mem = self._sim.smem if shared else self._sim.gmem
        i = int(idx)
        if shared:
            self._s.append(i)
        else:
            self._g.append(i * 4)
            self._sim._wrote(i, self.gid, atomic=True)
        old = float(mem[i])
        if op == "add":
            mem[i] = old + val
        elif op == "sub":
            mem[i] = old - val
        elif op == "max":
            mem[i] = max(old, val)
        elif op == "min":
            mem[i] = min(old, val)
        elif op == "exch":
            mem[i] = val
        elif op == "cas":
            # val is (compare, replacement) — CAS is the primitive the others build on
            cmp_, rep = val
            if old == cmp_:
                mem[i] = rep
        else:
            raise ValueError("unknown atomic op: %s" % op)
        self._alu += 1
        self._sim._atomics += 1
        return old

    def sload(self, widx):
        self._s.append(int(widx))
        return self._sim.smem[int(widx)]

    def sstore(self, widx, val):
        self._s.append(int(widx))
        self._sim.smem[int(widx)] = val

    def alu(self, n=1):
        self._alu += n

    # --- convenience ops the kernel may use; each one costs ALU like on hardware
    def exp(self, x):
        self._alu += 1
        return float(np.exp(x))

    def fma(self, a, b, c):
        self._alu += 1
        return a * b + c

    def rsqrt(self, x):
        self._alu += 1
        return 1.0 / float(np.sqrt(x))

    def salloc(self, n):
        """Reserve n words of __shared__ and return the base word index.

        Mirrors declaring `__shared__ float buf[n];`. The declaration belongs to
        the *block*, not to the thread: all 256 threads of a block executing the
        same declaration must get the same base, and it must be reserved once.
        So the k-th salloc() call of a block allocates, and every later thread
        reaching its own k-th call gets that same base back.
        """
        k = self._salloc_i
        self._salloc_i += 1
        return self._sim._smem_decl(k, n)

    def shared(self, n):
        """Alias of salloc(); reads better as `base = t.shared(TILE*TILE)`."""
        return self.salloc(n)

    def shfl_up(self, val, delta):
        """Warp shuffle: receive the value held `delta` lanes below.

        Must be used from a generator kernel as `got = yield t.shfl_up(v, d)`,
        because it is warp-synchronous — see ShflRequest.
        """
        return ShflRequest("up", val, delta)

    def shfl_down(self, val, delta):
        return ShflRequest("down", val, delta)

    def shfl_xor(self, val, mask):
        return ShflRequest("xor", val, mask)


# --- occupancy model -------------------------------------------------------
# Per-SM resource limits. These are the Ampere/Ada numbers; they are pinned here
# so the answer is the same on every machine, which is the whole point.
SM_REGS = 65536          # 32-bit registers in the SM register file
SM_SMEM = 49152          # bytes of shared memory available to blocks
SM_MAX_WARPS = 48        # resident warp slots
SM_MAX_BLOCKS = 16       # resident block slots
REG_GRAN = 256           # registers are allocated per warp in this granularity
MAX_REGS_PER_THREAD = 255


def occupancy(regs_per_thread, smem_bytes_per_block, block_size):
    """Resident warps per SM, and which resource is the limiter.

    This is the arithmetic `nvcc --ptxas-options=-v` plus the occupancy
    calculator do: round the register allocation up to the granularity, see how
    many blocks each resource allows, take the smallest, and report what bound
    it. Anything above MAX_REGS_PER_THREAD cannot be held in registers at all —
    the compiler spills the excess to local memory, which is really global.
    """
    warps_per_block = -(-block_size // WARP)      # ceil
    spilled = max(0, regs_per_thread - MAX_REGS_PER_THREAD)
    regs = min(regs_per_thread, MAX_REGS_PER_THREAD)

    regs_per_warp = -(-(regs * WARP) // REG_GRAN) * REG_GRAN
    by_reg = SM_REGS // (regs_per_warp * warps_per_block) if regs_per_warp else SM_MAX_BLOCKS
    by_smem = SM_SMEM // smem_bytes_per_block if smem_bytes_per_block else SM_MAX_BLOCKS
    by_warp = SM_MAX_WARPS // warps_per_block if warps_per_block else SM_MAX_BLOCKS
    blocks = max(0, min(by_reg, by_smem, by_warp, SM_MAX_BLOCKS))

    limiter = min((by_reg, "registers"), (by_smem, "shared memory"),
                  (by_warp, "warp slots"), (SM_MAX_BLOCKS, "block slots"))[1]
    warps = blocks * warps_per_block
    return {"blocks_per_sm": blocks, "warps_per_sm": warps,
            "occupancy": warps / SM_MAX_WARPS, "limiter": limiter,
            "regs_per_warp": regs_per_warp, "spilled_regs": spilled}


def latency_hiding(warps_per_sm, ilp, mem_latency=CYC_MEM, issue_cycles=4):
    """Cycles to cover one memory latency, given occupancy AND per-thread ILP.

    Volkov's point: occupancy is not the only way to hide latency. A kernel with
    few warps but several independent operations in flight per thread can cover
    the same latency as a kernel with many warps and no ILP. Both terms multiply.
    """
    in_flight = max(1, warps_per_sm) * max(1, ilp)
    covered = in_flight * issue_cycles
    return {"in_flight": in_flight, "covered_cycles": covered,
            "stall_cycles": max(0, mem_latency - covered),
            "hides_latency": covered >= mem_latency}


def _dims(v):
    """Accept 4 as (4, 1) and (4, 8) as itself — a scalar launch is a 1D launch."""
    if isinstance(v, (tuple, list)):
        x = int(v[0])
        y = int(v[1]) if len(v) > 1 else 1
        return x, y
    return int(v), 1


class GPU:
    def __init__(self, gmem_size, smem_size=0):
        self.gmem = np.zeros(gmem_size, dtype=np.float64)
        self.smem = np.zeros(max(smem_size, 1), dtype=np.float64)
        self._smem_next = 0
        self._smem_decls = []
        # address -> {"threads": set of gids that wrote it, "atomic": bool}
        self._writes = {}
        self._atomics = 0

    def _wrote(self, addr, gid, atomic):
        """Record a write so a lost-update race can be detected deterministically.

        Real hardware loses updates when several threads read-modify-write the
        same address without an atomic; here every thread runs to completion in
        turn, so the arithmetic would silently come out right. Rather than fake
        nondeterminism, the simulator reports the hazard itself: any address
        written by more than one thread with at least one NON-atomic write is a
        race, and `races` counts those addresses.
        """
        w = self._writes.get(addr)
        if w is None:
            self._writes[addr] = {"threads": {gid}, "nonatomic": not atomic}
        else:
            w["threads"].add(gid)
            if not atomic:
                w["nonatomic"] = True

    def _smem_decl(self, k, n):
        """Return the base of this block's k-th __shared__ declaration.

        Allocated once per block (by whichever thread reaches it first) and
        replayed to every other thread, then reset at the block boundary — which
        is what per-block shared memory means on hardware.
        """
        if k < len(self._smem_decls):
            base, size = self._smem_decls[k]
            if size != n:
                raise ValueError(
                    f"__shared__ declaration {k} is {size} words for one thread and {n} for another; "
                    "shared allocations are per-block and must be uniform across the block")
            return base
        base = self._smem_next
        if base + n > len(self.smem):
            raise ValueError(f"__shared__ overflow: need {base + n} words, have {len(self.smem)}")
        self._smem_next = base + n
        self._smem_decls.append((base, n))
        return base

    def launch(self, kernel, grid, block, *args):
        """Run kernel(thread, *args) over grid*block threads; return metrics.

        `grid` and `block` are ints for a 1D launch, or (x, y) tuples for a 2D
        one — the shape real tiling kernels (transpose, tiled matmul, flash
        attention tiles) are written in. A scalar stays exactly as before.

        Two kernel styles are auto-detected (no API change for callers):

          - plain function `def kernel(t, *args): ...` — executed start-to-
            finish immediately, exactly as before. No barrier is possible
            (there is nowhere to pause), so this style must not contain a
            cross-thread dependency that needs `__syncthreads()`.

          - generator function `def kernel(t, *args): ... yield ...` (used by
            the CUDA-C interpreter in `mlsys.sim.cuda_c` to represent
            `__syncthreads()`) — driven in lockstep, round-robin, across every
            thread of a block: each thread is stepped until it yields (hits a
            barrier) or finishes; once every thread of the block has reached
            that point the barrier releases and the next phase begins. This
            gives real cross-thread barrier semantics — a thread can observe
            writes any other thread in the block made before the barrier —
            without real OS/hardware threads.

            A bare `yield` is a barrier. Yielding a `ShflRequest` (from
            `t.shfl_up/down/xor`) is a warp shuffle: it is collected across the
            whole warp at that same program point and each lane is resumed with
            its own result, which is what makes `__shfl_up_sync` meaningful.
        """
        is_barrier_kernel = inspect.isgeneratorfunction(kernel)
        self._writes = {}
        self._atomics = 0
        gx, gy = _dims(grid)
        bx, by = _dims(block)
        block_n = bx * by
        threads = [[Thread(self, tid % bx, bid % gx, bx, gx,
                           tid_y=tid // bx, bid_y=bid // gx, bdim_y=by, gdim_y=gy)
                    for tid in range(block_n)]
                   for bid in range(gx * gy)]
        for bid in range(gx * gy):
            # __shared__ memory is per-block on real hardware: private to the
            # block, not carried over from whatever the previous block left
            # behind. Reset it before each block starts.
            self.smem[:] = 0
            self._smem_next = 0
            self._smem_decls = []
            if is_barrier_kernel:
                blk_threads = threads[bid]
                gens = [kernel(th, *args) for th in blk_threads]
                active = list(range(block_n))
                send = [None] * block_n
                while active:
                    still_running, pending = [], {}
                    for tid in active:
                        try:
                            y = gens[tid].send(send[tid])
                            send[tid] = None
                            still_running.append(tid)  # yielded at a barrier or a shuffle
                            if isinstance(y, ShflRequest):
                                pending[tid] = y
                        except StopIteration:
                            pass  # this thread has finished the kernel
                    if pending:
                        self._resolve_shuffles(blk_threads, pending, send)
                    active = still_running
            else:
                for tid in range(block_n):
                    kernel(threads[bid][tid], *args)

        return self._metrics(threads)

    @staticmethod
    def _resolve_shuffles(blk_threads, pending, send):
        """Serve one warp-synchronous shuffle step.

        Every lane that yielded a ShflRequest at this program point publishes its
        value; each requesting lane then reads the lane it asked for. A lane whose
        source is inactive (outside the warp, or not participating) keeps its own
        value — the documented behaviour of `__shfl_*_sync` for an out-of-range
        source.
        """
        by_warp = {}
        for tid, req in pending.items():
            by_warp.setdefault(tid // WARP, {})[tid] = req
        for w, members in by_warp.items():
            vals = {tid % WARP: r.value for tid, r in members.items()}
            for tid, r in members.items():
                lane = tid % WARP
                if r.kind == "up":
                    src = lane - r.delta
                elif r.kind == "down":
                    src = lane + r.delta
                else:                       # xor (butterfly)
                    src = lane ^ r.delta
                send[tid] = vals.get(src, r.value) if 0 <= src < WARP else r.value
                blk_threads[tid]._alu += 1  # a shuffle is an instruction, not free

    def _metrics(self, threads):
        flat = [t for blk in threads for t in blk]
        transactions = smem_waves = divergences = alu_ops = 0
        for w0 in range(0, len(flat), WARP):
            warp = flat[w0:w0 + WARP]
            # divergence: threads in a warp that issued different #accesses
            glens = {len(t._g) for t in warp}
            if len(glens) > 1:
                divergences += 1
            # coalescing: per access-step, distinct 128B segments touched by the warp
            for k in range(max((len(t._g) for t in warp), default=0)):
                segs = {t._g[k] // SEGMENT for t in warp if k < len(t._g)}
                transactions += len(segs)
            # shared-memory bank conflicts: per step, max distinct words per bank
            for k in range(max((len(t._s) for t in warp), default=0)):
                per_bank = {}
                for t in warp:
                    if k < len(t._s):
                        per_bank.setdefault(t._s[k] % BANKS, set()).add(t._s[k])
                smem_waves += max((len(s) for s in per_bank.values()), default=0)
            alu_ops += max((t._alu for t in warp), default=0)

        # Memory INSTRUCTIONS issued, which is what vectorising actually cuts: a
        # float4 load moves the same bytes as four float loads and needs the same
        # 128-byte transactions, but it is one instruction instead of four.
        mem_insts = sum(len(t._g) for t in flat)
        smem_insts = sum(len(t._s) for t in flat)
        races = sum(1 for w in self._writes.values()
                    if len(w["threads"]) > 1 and w["nonatomic"])
        cycles = transactions * CYC_MEM + smem_waves * CYC_SMEM + alu_ops * CYC_ALU
        warps = (len(flat) + WARP - 1) // WARP
        return {"result": self.gmem, "transactions": transactions,
                "smem_waves": smem_waves, "divergences": divergences,
                "cycles": cycles, "warps": warps,
                "races": races, "atomics": self._atomics,
                "mem_insts": mem_insts, "smem_insts": smem_insts,
                "cycles_per_warp": cycles / warps if warps else 0}
