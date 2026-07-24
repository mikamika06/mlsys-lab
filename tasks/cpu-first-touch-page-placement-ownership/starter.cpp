#include "sol.hpp"

void first_touch_owner(const int* thread_of_access, const int* page_of_access, int n,
                        const int* node_of_thread, int num_threads,
                        int num_pages, int* owner_of_page) {
    // your code here
    (void)thread_of_access;
    (void)page_of_access;
    (void)n;
    (void)node_of_thread;
    (void)num_threads;
    for (int p = 0; p < num_pages; ++p) owner_of_page[p] = -1;
}
