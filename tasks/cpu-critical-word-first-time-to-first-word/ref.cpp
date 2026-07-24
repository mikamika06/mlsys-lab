#include "sol.hpp"

long time_to_word(int line_words, int start_word, int target_word,
                   int base_latency, int cycles_per_word) {
    int diff = target_word - start_word;
    int position = ((diff % line_words) + line_words) % line_words;
    return (long)base_latency + (long)position * (long)cycles_per_word;
}
