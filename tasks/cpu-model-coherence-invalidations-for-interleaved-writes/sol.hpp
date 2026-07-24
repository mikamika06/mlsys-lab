#pragma once

// One write in a multi-core trace: core `core` writes to byte address
// `addr`.
struct WriteEvent {
    int core;
    long addr;
};

// A simplified write-invalidate cache-coherence model over 64-byte
// lines. Each line has a set of cores currently holding a valid cached
// copy of it (starts empty for every line). For each event in `trace`,
// in order:
//   - let `line = addr / 64`;
//   - every OTHER core currently in that line's owner set must have
//     its copy invalidated before this write can proceed -- count one
//     invalidation per such core;
//   - after the write, that line's owner set becomes just {core} (this
//     write's core is now the sole holder; everyone else's copy is
//     gone).
// Returns the total invalidation count summed over the whole trace.
long count_invalidations(const WriteEvent* trace, int n);
