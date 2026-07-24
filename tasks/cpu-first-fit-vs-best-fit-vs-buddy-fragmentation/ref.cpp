#include <algorithm>
#include <vector>
#include "sol.hpp"

namespace {
constexpr int ARENA_BYTES = 256;
constexpr int BUDDY_MIN = 16;

struct Block { int offset; int size; bool free; };

// Address-ordered free list shared by first-fit and best-fit.
struct FreeListAlloc {
    std::vector<Block> blocks;  // kept sorted by offset
    bool best_fit;
    explicit FreeListAlloc(bool bf) : best_fit(bf) {
        blocks.push_back({0, ARENA_BYTES, true});
    }
    int alloc(int size) {
        int chosen = -1;
        for (int i = 0; i < (int)blocks.size(); i++) {
            if (!blocks[i].free || blocks[i].size < size) continue;
            if (chosen < 0) {
                chosen = i;
                if (!best_fit) break;  // first-fit: take the first candidate
                continue;
            }
            if (best_fit && blocks[i].size < blocks[chosen].size) chosen = i;
        }
        if (chosen < 0) return -1;
        Block b = blocks[chosen];
        int offset = b.offset;
        if (b.size > size) {
            blocks[chosen].size = size;
            blocks[chosen].free = false;
            blocks.insert(blocks.begin() + chosen + 1, Block{b.offset + size, b.size - size, true});
        } else {
            blocks[chosen].free = false;
        }
        return offset;
    }
    void free(int offset) {
        if (offset < 0) return;
        for (int i = 0; i < (int)blocks.size(); i++) {
            if (blocks[i].offset != offset || blocks[i].free) continue;
            blocks[i].free = true;
            if (i + 1 < (int)blocks.size() && blocks[i + 1].free) {
                blocks[i].size += blocks[i + 1].size;
                blocks.erase(blocks.begin() + i + 1);
            }
            if (i - 1 >= 0 && blocks[i - 1].free) {
                blocks[i - 1].size += blocks[i].size;
                blocks.erase(blocks.begin() + i);
            }
            return;
        }
    }
    void fragmentation(int* total_free, int* largest_free) const {
        int tf = 0, lf = 0;
        for (const auto& b : blocks)
            if (b.free) { tf += b.size; lf = std::max(lf, b.size); }
        *total_free = tf; *largest_free = lf;
    }
};

// Power-of-two buddy allocator, 5 levels: block sizes 256,128,64,32,16.
struct BuddyAlloc {
    static const int NUM_LEVELS = 5;
    std::vector<int> freeList[NUM_LEVELS];  // block indices, per level
    int levelOfAlloc[ARENA_BYTES / BUDDY_MIN];
    BuddyAlloc() {
        for (int i = 0; i < ARENA_BYTES / BUDDY_MIN; i++) levelOfAlloc[i] = -1;
        freeList[0].push_back(0);
    }
    static int blockSize(int level) { return ARENA_BYTES >> level; }
    int alloc(int size) {
        int need = size < BUDDY_MIN ? BUDDY_MIN : size;
        int targetLevel = -1;
        for (int l = NUM_LEVELS - 1; l >= 0; l--)
            if (blockSize(l) >= need) { targetLevel = l; break; }
        if (targetLevel < 0) return -1;

        int foundLevel = -1;
        for (int l = targetLevel; l >= 0; l--)
            if (!freeList[l].empty()) { foundLevel = l; break; }
        if (foundLevel < 0) return -1;

        std::sort(freeList[foundLevel].begin(), freeList[foundLevel].end());
        int idx = freeList[foundLevel].front();
        freeList[foundLevel].erase(freeList[foundLevel].begin());
        for (int l = foundLevel; l < targetLevel; l++) {
            int left = 2 * idx, right = 2 * idx + 1;
            freeList[l + 1].push_back(right);
            idx = left;
        }
        int offset = idx * blockSize(targetLevel);
        levelOfAlloc[offset / BUDDY_MIN] = targetLevel;
        return offset;
    }
    void free(int offset) {
        if (offset < 0) return;
        int finestIdx = offset / BUDDY_MIN;
        int level = levelOfAlloc[finestIdx];
        if (level < 0) return;
        levelOfAlloc[finestIdx] = -1;
        int idx = offset / blockSize(level);
        while (level > 0) {
            int buddyIdx = idx ^ 1;
            auto& fl = freeList[level];
            auto it = std::find(fl.begin(), fl.end(), buddyIdx);
            if (it == fl.end()) break;
            fl.erase(it);
            idx /= 2;
            level -= 1;
        }
        freeList[level].push_back(idx);
    }
    void fragmentation(int* total_free, int* largest_free) const {
        int tf = 0, lf = 0;
        for (int l = 0; l < NUM_LEVELS; l++) {
            int sz = blockSize(l);
            tf += (int)freeList[l].size() * sz;
            if (!freeList[l].empty()) lf = std::max(lf, sz);
        }
        *total_free = tf; *largest_free = lf;
    }
};
}  // namespace

void fragmentation_after_trace(const int* op_kind, const int* op_arg, int num_ops, int* out) {
    FreeListAlloc ff(false), bf(true);
    BuddyAlloc bud;
    std::vector<int> ff_h, bf_h, bud_h;

    for (int i = 0; i < num_ops; i++) {
        if (op_kind[i] == 0) {
            int size = op_arg[i];
            ff_h.push_back(ff.alloc(size));
            bf_h.push_back(bf.alloc(size));
            bud_h.push_back(bud.alloc(size));
        } else {
            int ai = op_arg[i];
            ff.free(ff_h[ai]);
            bf.free(bf_h[ai]);
            bud.free(bud_h[ai]);
        }
    }

    int tf, lf;
    ff.fragmentation(&tf, &lf); out[0] = tf - lf;
    bf.fragmentation(&tf, &lf); out[1] = tf - lf;
    bud.fragmentation(&tf, &lf); out[2] = tf - lf;
}
