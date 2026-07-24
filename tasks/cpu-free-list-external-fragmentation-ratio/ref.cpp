#include <list>
#include <vector>
#include "sol.hpp"

namespace {
struct Block { long offset, size; bool free; };
}

double external_fragmentation_ratio(long heap_bytes,
                                     const int* op_types, const int* op_sizes, const int* op_ids,
                                     int num_ops) {
    std::list<Block> blocks;
    blocks.push_back({0, heap_bytes, true});

    std::vector<long> alloc_offset(num_ops, -1);

    for (int i = 0; i < num_ops; i++) {
        if (op_types[i] == 0) {  // ALLOC
            long req = op_sizes[i];
            for (auto it = blocks.begin(); it != blocks.end(); ++it) {
                if (it->free && it->size >= req) {
                    long off = it->offset;
                    if (it->size == req) {
                        it->free = false;
                    } else {
                        long leftover_off = it->offset + req;
                        long leftover_size = it->size - req;
                        it->size = req;
                        it->free = false;
                        blocks.insert(std::next(it), Block{leftover_off, leftover_size, true});
                    }
                    alloc_offset[i] = off;
                    break;
                }
            }
        } else {  // FREE
            long off = alloc_offset[op_ids[i]];
            for (auto it = blocks.begin(); it != blocks.end(); ++it) {
                if (it->offset == off) {
                    it->free = true;
                    auto nxt = std::next(it);
                    if (nxt != blocks.end() && nxt->free) {
                        it->size += nxt->size;
                        blocks.erase(nxt);
                    }
                    if (it != blocks.begin()) {
                        auto prv = std::prev(it);
                        if (prv->free) {
                            prv->size += it->size;
                            blocks.erase(it);
                        }
                    }
                    break;
                }
            }
        }
    }

    long total_free = 0, largest_free = 0;
    for (const auto& b : blocks) {
        if (b.free) {
            total_free += b.size;
            if (b.size > largest_free) largest_free = b.size;
        }
    }
    if (total_free == 0) return 0.0;
    return (double)(total_free - largest_free) / (double)total_free;
}
