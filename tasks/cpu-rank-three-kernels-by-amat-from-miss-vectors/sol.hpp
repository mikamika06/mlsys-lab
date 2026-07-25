#pragma once

// Average memory access time (AMAT) under a 2-level cache hierarchy,
// with fixed hit/penalty costs L1_HIT=1, L2_HIT=10, MEM_PENALTY=100
// cycles:
//
//   L1_miss_rate  = l1_misses[k] / accesses[k]
//   L2_miss_rate  = l2_misses[k] / l1_misses[k]   (LOCAL miss rate: the
//                                                   fraction of L1 MISSES
//                                                   that also miss L2)
//   AMAT[k] = L1_HIT + L1_miss_rate * (L2_HIT + L2_miss_rate * MEM_PENALTY)
//
// For 3 kernels (arrays indexed 0..2), compute AMAT[k] for each and
// write them into amat_out[0..3). Write the 3 kernel ids into
// rank_out[0..3), sorted by ASCENDING amat (fastest kernel first).
void rank_by_amat(const long* accesses, const long* l1_misses, const long* l2_misses,
                   double* amat_out, int* rank_out);
