#pragma once

// A page-walk cache (PWC) has one independent, fully-associative LRU
// cache per page-table level -- level 0 = PML4, 1 = PDPT, 2 = PD,
// 3 = PT -- each remembering the table keys it has resolved recently.
// `cap[i]` is level i's capacity (max distinct keys it can hold at
// once).
//
// Process `num_addrs` virtual-address translations in order. Address
// j's four per-level table keys are `keys[j*4 + i]` for `i` in `0..3`
// (row-major, 4 keys per address). For each address, walk levels 0..3
// in order; at level i:
//   - if `keys[j*4+i]` is already resident in level i's PWC (inserted
//     by walking an earlier address), that level costs `hit_cycles`,
//     and the key becomes level i's most-recently-used;
//   - otherwise it costs `miss_cycles`, and the key is inserted as
//     level i's most-recently-used, evicting that level's
//     least-recently-used key first if the level's cache already holds
//     `cap[i]` keys.
// After all 4 levels are resolved, the final data access always costs
// `data_cycles` (this model assumes the data itself is always resident;
// the walk overhead is the point).
//
// Return the TOTAL cycles summed over all `num_addrs` translations.
long page_walk_cycles(const int* keys, int num_addrs, const int* cap,
                       long hit_cycles, long miss_cycles, long data_cycles);
