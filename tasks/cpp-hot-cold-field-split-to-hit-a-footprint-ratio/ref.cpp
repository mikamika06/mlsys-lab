#include "sol.hpp"

int split_struct(const int* fields, const int* is_hot, int n,
                  int* hot_out, int* cold_out) {
    int hot_count = 0, cold_count = 0;
    for (int i = 0; i < n; i++) {
        if (is_hot[i]) hot_out[hot_count++] = fields[i];
        else cold_out[cold_count++] = fields[i];
    }

    // Sort hot_out in DESCENDING order of size. Since every field's natural
    // alignment equals its own size here, placing the largest (most
    // alignment-demanding) fields first and shrinking from there never
    // needs padding BEFORE a field of size s once every field of size >= s
    // has already been placed at an offset that's already a multiple of s
    // -- this greedy order is optimal for this "alignment == size" case.
    for (int i = 0; i < hot_count; i++) {
        int max_j = i;
        for (int j = i + 1; j < hot_count; j++) {
            if (hot_out[j] > hot_out[max_j]) max_j = j;
        }
        int tmp = hot_out[i];
        hot_out[i] = hot_out[max_j];
        hot_out[max_j] = tmp;
    }

    return hot_count;
}
