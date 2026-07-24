#include "sol.hpp"

void first_touch_owner(const int* thread_of_access, const int* page_of_access, int n,
                        const int* node_of_thread, int num_threads,
                        int num_pages, int* owner_of_page) {
    (void)num_threads;
    for (int p = 0; p < num_pages; ++p) owner_of_page[p] = -1;

    for (int i = 0; i < n; ++i) {
        int page = page_of_access[i];
        if (owner_of_page[page] == -1) {
            int thread = thread_of_access[i];
            owner_of_page[page] = node_of_thread[thread];
        }
    }
}
