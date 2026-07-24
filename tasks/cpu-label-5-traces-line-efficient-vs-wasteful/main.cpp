#include <cstdio>
#include <vector>
#include "sol.hpp"

int main() {
    const int ELEM = 4, LINE = 64;

    // Trace 1: contiguous sweep, stride == elem_bytes -- every fetched
    // byte gets used.
    std::vector<long> t1;
    for (int i = 0; i < 256; i++) t1.push_back((long)i * 4);

    // Trace 2: large stride (== line_bytes) -- one 4-byte element used
    // out of every 64-byte line fetched.
    std::vector<long> t2;
    for (int i = 0; i < 64; i++) t2.push_back((long)i * 64);

    // Trace 3: medium stride (16 bytes) -- 4 of every 16 bytes used, so
    // 4 elements used per 64-byte line (16 bytes of 64 fetched).
    std::vector<long> t3;
    for (int i = 0; i < 256; i++) t3.push_back((long)i * 16);

    // Trace 4: stride 8 bytes -- every other element, exactly half of
    // each fetched line's bytes get used (boundary case at 0.5).
    std::vector<long> t4;
    for (int i = 0; i < 256; i++) t4.push_back((long)i * 8);

    // Trace 5: heavy reuse -- 200 accesses cycling through 4 addresses
    // that all sit inside a single 64-byte line.
    std::vector<long> t5;
    for (int i = 0; i < 200; i++) t5.push_back((long)(i % 4) * 4);

    int l1 = classify_trace(t1.data(), (int)t1.size(), ELEM, LINE);
    int l2 = classify_trace(t2.data(), (int)t2.size(), ELEM, LINE);
    int l3 = classify_trace(t3.data(), (int)t3.size(), ELEM, LINE);
    int l4 = classify_trace(t4.data(), (int)t4.size(), ELEM, LINE);
    int l5 = classify_trace(t5.data(), (int)t5.size(), ELEM, LINE);

    printf("labels=%d,%d,%d,%d,%d\n", l1, l2, l3, l4, l5);
    return 0;
}
