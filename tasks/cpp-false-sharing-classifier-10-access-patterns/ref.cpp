#include "sol.hpp"
#include <cstddef>

namespace {

struct Access {
    long byte_offset;
    bool is_write;
};

long cache_line(long byte_offset) { return byte_offset / 64; }

long off_items(int i) {
    return (long)(offsetof(ThreadState, items) + i * sizeof(double));
}
long off_padding(int i) {
    return (long)(offsetof(ThreadState, padding) + i * sizeof(long long));
}

}  // namespace

std::pair<std::vector<bool>, long> classify_false_sharing() {
    struct Pattern { Access a, b; };

    std::vector<Pattern> patterns = {
        {{(long)offsetof(ThreadState, id), true},
         {(long)offsetof(ThreadState, read_count), true}},
        {{(long)offsetof(ThreadState, id), true},
         {(long)offsetof(ThreadState, write_count), false}},
        {{(long)offsetof(ThreadState, read_count), false},
         {(long)offsetof(ThreadState, write_count), false}},
        {{off_items(4), true}, {off_padding(0), true}},
        {{off_items(4), true}, {off_padding(1), true}},
        {{off_padding(0), true}, {off_padding(1), true}},
        {{(long)offsetof(ThreadState, local_sum), true},
         {(long)offsetof(ThreadState, local_flag), true}},
        {{off_items(0), true},
         {(long)offsetof(ThreadState, local_sum), false}},
        {{off_padding(1), true},
         {(long)offsetof(ThreadState, local_flag), false}},
        {{off_items(0), false}, {off_items(1), true}},
    };

    std::vector<bool> labels;
    labels.reserve(patterns.size());
    for (const auto& p : patterns) {
        bool same_line = cache_line(p.a.byte_offset) == cache_line(p.b.byte_offset);
        bool any_write = p.a.is_write || p.b.is_write;
        labels.push_back(same_line && any_write);
    }

    return {labels, (long)sizeof(ThreadState)};
}
