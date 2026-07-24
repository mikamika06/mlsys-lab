#pragma once

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// Modeled first-touch NUMA placement: physical pages are allocated lazily,
// and under the default Linux "first-touch" policy a page's physical memory
// is placed on whichever NUMA node the thread that accesses it FIRST happens
// to run on. Every later access from a different node reads/writes that page
// remotely (slower), but it never moves the page.
//
// Given a trace of `n` accesses, in order:
//   - thread_of_access[i]: which thread performs the i-th access
//   - page_of_access[i]:   which page that access touches
// and a thread -> NUMA-node map:
//   - node_of_thread[t] for t in [0, num_threads)
// compute, for every page id in [0, num_pages), the NUMA node that owns it:
// the node of whichever thread's access to that page occurs EARLIEST in the
// trace. A page that is never touched by any access has no owner -- report
// it as -1.
//
// Write the result into `owner_of_page[0 .. num_pages)` (caller-allocated).
// ============================================================================
void first_touch_owner(const int* thread_of_access, const int* page_of_access, int n,
                        const int* node_of_thread, int num_threads,
                        int num_pages, int* owner_of_page);
