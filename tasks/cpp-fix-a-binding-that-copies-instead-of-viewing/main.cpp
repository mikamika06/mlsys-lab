#include <cstdio>
#include "sol.hpp"

int main() {
    const int N = 6;
    const int header_size = 24;   // e.g. sizeof({long, long, void*}) on LP64
    const int total = header_size + N * (int)sizeof(double);

    unsigned char* buf = new unsigned char[total];
    for (int i = 0; i < header_size; i++) buf[i] = (unsigned char)(i + 1);  // filler header bytes

    double* payload = reinterpret_cast<double*>(buf + header_size);
    for (int i = 0; i < N; i++) payload[i] = i * 1.5;                      // deterministic payload

    double* view = view_payload(buf, header_size, N);

    // 1. Reading through the view must see the current payload values.
    printf("read:");
    for (int i = 0; i < N; i++) printf(" %.3f", view[i]);
    printf("\n");

    // 2. Writing through the view, then re-reading directly out of `buf`,
    //    must see the write -- this is the actual zero-copy test.
    for (int i = 0; i < N; i++) view[i] = view[i] * 2.0 + 1.0;
    double* direct = reinterpret_cast<double*>(buf + header_size);
    printf("direct-after-write:");
    for (int i = 0; i < N; i++) printf(" %.3f", direct[i]);
    printf("\n");

    // 3. A real zero-copy view aliases buf's own memory exactly.
    printf("shares_memory=%d\n", (view == direct) ? 1 : 0);

    delete[] buf;
    return 0;
}
