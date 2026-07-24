#include "sol.hpp"

namespace {
const int SIZE[4] = {4, 4, 4, 8};
const int AOS_OFFSET[4] = {0, 4, 8, 12};
const int RECORD_SIZE = 20;
const int BLOCK_W = 8;
const int BLOCK_SIZE = BLOCK_W * RECORD_SIZE; // 160
const int AOSOA_SUBOFF[4] = {0, BLOCK_W * 4, BLOCK_W * 4 * 2, BLOCK_W * 4 * 3}; // 0,32,64,96

void touch_range(long start, int len) {
    for (int b = 0; b < len; b++) touch(start + b);
}
} // namespace

void emit_access(Layout layout, int n, int field_idx) {
    if (layout == Layout::AoS) {
        for (int i = 0; i < n; i++) {
            if (field_idx == -1) {
                touch_range(static_cast<long>(i) * RECORD_SIZE, RECORD_SIZE);
            } else {
                touch_range(static_cast<long>(i) * RECORD_SIZE + AOS_OFFSET[field_idx], SIZE[field_idx]);
            }
        }
    } else if (layout == Layout::SoA) {
        long base[4];
        base[0] = 0;
        for (int f = 1; f < 4; f++) base[f] = base[f - 1] + static_cast<long>(SIZE[f - 1]) * n;
        for (int i = 0; i < n; i++) {
            if (field_idx == -1) {
                for (int f = 0; f < 4; f++) {
                    touch_range(base[f] + static_cast<long>(i) * SIZE[f], SIZE[f]);
                }
            } else {
                touch_range(base[field_idx] + static_cast<long>(i) * SIZE[field_idx], SIZE[field_idx]);
            }
        }
    } else { // AoSoA
        for (int i = 0; i < n; i++) {
            int b = i / BLOCK_W;
            int p = i % BLOCK_W;
            long block_start = static_cast<long>(b) * BLOCK_SIZE;
            if (field_idx == -1) {
                for (int f = 0; f < 4; f++) {
                    touch_range(block_start + AOSOA_SUBOFF[f] + static_cast<long>(p) * SIZE[f], SIZE[f]);
                }
            } else {
                touch_range(block_start + AOSOA_SUBOFF[field_idx] + static_cast<long>(p) * SIZE[field_idx],
                             SIZE[field_idx]);
            }
        }
    }
}
