#include <cstdio>
#include "sol.hpp"

// FIXED driver. 10 writes across 3 threads, line_bytes=64.
//   line 0 (0..63):   thread0 @4,  thread1 @60   -> 2 threads, 2 addrs -> falsely shared
//   line 1 (64..127): thread0 @64, thread0 @68   -> 1 thread only     -> not shared
//   line 2 (128..191):thread0 @128,thread1 @128  -> same address      -> TRUE sharing, not false
//   line 3 (192..255):thread2 @200,thread0 @210,thread1 @220 -> falsely shared
//   line 4 (256..319):thread2 @300                -> 1 thread only    -> not shared
int main() {
    const int n = 10;
    long addrs[n]     = {4, 60, 64, 68, 128, 128, 200, 210, 220, 300};
    int thread_id[n]  = {0,  1,  0,  0,   0,   1,   2,   0,   1,   2};
    const int line_bytes = 64;

    long out[n];
    for (int i = 0; i < n; i++) out[i] = -1;  // sentinel: an empty starter leaves this untouched

    int count = find_falsely_shared_lines(addrs, thread_id, n, line_bytes, out);

    printf("count=%d\n", count);
    for (int i = 0; i < count; i++) printf("%ld ", out[i]);
    printf("\n");
    return 0;
}
