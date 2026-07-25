#pragma once

// Two standard prefetcher quality metrics, given 3 raw counts already
// measured by the harness from running the SAME access trace once
// without prefetching and once with a next-line prefetcher enabled:
//
//   baseline_misses    -- misses with NO prefetching at all.
//   total_prefetches   -- number of prefetch requests the prefetcher
//                          issued.
//   useful_prefetches  -- of those, how many were actually consumed by a
//                          later access before being evicted (i.e. really
//                          turned a would-be miss into a hit).
//
// coverage = useful_prefetches / baseline_misses
//   -- what fraction of the misses that WOULD have happened did the
//      prefetcher actually eliminate.
// accuracy = useful_prefetches / total_prefetches
//   -- what fraction of the prefetches issued were actually worth doing
//      (the rest were wasted bandwidth/cache space).
//
// Write coverage into *coverage_out and accuracy into *accuracy_out.
void compute_coverage_accuracy(long baseline_misses, long total_prefetches, long useful_prefetches,
                                double* coverage_out, double* accuracy_out);
