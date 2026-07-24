#pragma once

// A classic power-of-two buddy allocator over a fixed ARENA_BYTES arena,
// with minimum block size MIN_BLOCK_BYTES. Levels run from 0 (one single
// ARENA_BYTES block) to NUM_LEVELS-1 (MAX_BLOCKS blocks of MIN_BLOCK_BYTES
// each); level l holds blocks of size (ARENA_BYTES >> l).
struct BuddyAllocator {
    static const int ARENA_BYTES = 256;
    static const int MIN_BLOCK_BYTES = 16;
    static const int NUM_LEVELS = 5;   // block sizes: 256, 128, 64, 32, 16
    static const int MAX_BLOCKS = 16;  // blocks at the finest level (256/16)

    // freeBlock[l][i]: true iff the block at level l, index i is currently
    // a whole, unallocated, unsplit free block.
    bool freeBlock[NUM_LEVELS][MAX_BLOCKS];

    // levelOfAlloc[offset / MIN_BLOCK_BYTES]: the level a LIVE allocation
    // starting at that finest-granularity slot was made at, or -1 if that
    // slot is not the start of a live allocation. free() uses this to
    // recover the block's size from just its offset.
    int levelOfAlloc[MAX_BLOCKS];

    // Starts with the whole arena as one free block (freeBlock[0][0] = true,
    // everything else false; levelOfAlloc all -1).
    BuddyAllocator();

    // Rounds `size` up to the smallest power-of-two block size that fits it
    // (at least MIN_BLOCK_BYTES), finds the smallest already-available free
    // block that is at least that big, and SPLITS it down (marking each
    // split-off buddy as newly free at the next level) until a block of
    // exactly the right size exists and can be marked allocated. Returns
    // that block's byte offset within the arena, or -1 if no block is big
    // enough or available.
    int alloc(int size);

    // Returns the (live) allocation at `offset` to the free list, then
    // MERGES it back together with its buddy -- repeatedly, as far up the
    // levels as possible -- whenever that buddy is ALSO a whole free block.
    // A no-op if `offset` is not the start of a live allocation.
    void free(int offset);
};

inline int buddyBlockSize(int level) { return BuddyAllocator::ARENA_BYTES >> level; }
inline int buddyNumBlocks(int level) { return 1 << level; }
