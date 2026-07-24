#include <cstdio>
#include "sol.hpp"

int struct_size(const int* sizes, int n) {
    int offset = 0;
    int max_align = 1;
    for (int i = 0; i < n; i++) {
        int s = sizes[i];
        if (s > max_align) max_align = s;
        int rem = offset % s;
        if (rem != 0) offset += (s - rem);
        offset += s;
    }
    int rem = offset % max_align;
    if (rem != 0) offset += (max_align - rem);
    return offset;
}

static void run_case(const int* fields, const int* is_hot, int n) {
    int hot_out[16], cold_out[16];
    for (int i = 0; i < 16; i++) { hot_out[i] = -1; cold_out[i] = -1; }

    int hot_count = split_struct(fields, is_hot, n, hot_out, cold_out);
    if (hot_count < 0) hot_count = 0;
    if (hot_count > 16) hot_count = 16;
    int cold_count = n - hot_count;
    if (cold_count < 0) cold_count = 0;
    if (cold_count > 16) cold_count = 16;

    printf("original=%d\n", struct_size(fields, n));
    printf("hot_count=%d hot=", hot_count);
    for (int i = 0; i < hot_count; i++) printf("%d ", hot_out[i]);
    printf("hot_size=%d\n", struct_size(hot_out, hot_count));
    printf("cold_count=%d cold=", cold_count);
    for (int i = 0; i < cold_count; i++) printf("%d ", cold_out[i]);
    printf("cold_size=%d\n", struct_size(cold_out, cold_count));
}

// FIXED driver. Two fat "Entity" structs, each with hot fields (touched in
// a tight per-frame loop) interleaved among cold fields (rarely touched) in
// a deliberately padding-unfriendly order.
int main() {
    // char, double, char, ptr, int, double, short, char, char, int, ptr, double
    const int fields1[]  = {1, 8, 1, 8, 4, 8, 2, 1, 1, 4, 8, 8};
    const int is_hot1[]  = {1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0};
    run_case(fields1, is_hot1, 12);

    // int, short, char, ptr, char, int, ptr, short
    const int fields2[] = {4, 2, 1, 8, 1, 4, 8, 2};
    const int is_hot2[] = {1, 0, 1, 0, 1, 1, 0, 0};
    run_case(fields2, is_hot2, 8);

    return 0;
}
