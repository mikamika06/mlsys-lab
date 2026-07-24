#include "sol.hpp"

BuddyAllocator::BuddyAllocator() {
    for (int l = 0; l < NUM_LEVELS; l++)
        for (int i = 0; i < MAX_BLOCKS; i++) freeBlock[l][i] = false;
    freeBlock[0][0] = true;
    for (int i = 0; i < MAX_BLOCKS; i++) levelOfAlloc[i] = -1;
}

int BuddyAllocator::alloc(int size) {
    int need = size < MIN_BLOCK_BYTES ? MIN_BLOCK_BYTES : size;

    int targetLevel = -1;
    for (int l = NUM_LEVELS - 1; l >= 0; l--) {
        if (buddyBlockSize(l) >= need) { targetLevel = l; break; }
    }
    if (targetLevel < 0) return -1;

    int foundLevel = -1, foundIdx = -1;
    for (int l = targetLevel; l >= 0 && foundLevel < 0; l--) {
        for (int i = 0; i < buddyNumBlocks(l); i++) {
            if (freeBlock[l][i]) { foundLevel = l; foundIdx = i; break; }
        }
    }
    if (foundLevel < 0) return -1;

    freeBlock[foundLevel][foundIdx] = false;
    int idx = foundIdx;
    for (int l = foundLevel; l < targetLevel; l++) {
        int leftChild = 2 * idx;
        int rightChild = 2 * idx + 1;
        freeBlock[l + 1][rightChild] = true;
        idx = leftChild;
    }

    int offset = idx * buddyBlockSize(targetLevel);
    levelOfAlloc[offset / MIN_BLOCK_BYTES] = targetLevel;
    return offset;
}

void BuddyAllocator::free(int offset) {
    int finestIdx = offset / MIN_BLOCK_BYTES;
    int level = levelOfAlloc[finestIdx];
    if (level < 0) return;
    levelOfAlloc[finestIdx] = -1;

    int idx = offset / buddyBlockSize(level);
    freeBlock[level][idx] = true;

    while (level > 0) {
        int buddyIdx = idx ^ 1;
        if (!freeBlock[level][buddyIdx]) break;
        freeBlock[level][idx] = false;
        freeBlock[level][buddyIdx] = false;
        idx = idx / 2;
        level = level - 1;
        freeBlock[level][idx] = true;
    }
}
