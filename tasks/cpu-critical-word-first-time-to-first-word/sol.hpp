#pragma once

// A cache-line fill transfers `line_words` words as a burst, one word
// every `cycles_per_word` cycles, with the very first transferred word
// landing `base_latency` cycles after the miss is issued. The burst does
// not have to start at word 0: it starts at `start_word` and then wraps
// circularly through every other word of the line exactly once --
// start_word, start_word+1, ..., line_words-1, 0, 1, ..., start_word-1 --
// before the fill is complete.
//
//   position = (target_word - start_word) mod line_words   -- in [0, line_words)
//   time     = base_latency + position * cycles_per_word
//
// Return the cycle count at which `target_word` becomes available. Note
// the true mathematical modulo: target_word can be less than start_word,
// and C++'s `%` returns a NEGATIVE result for a negative left operand --
// position must still land in [0, line_words).
//
// A memory system WITHOUT critical-word-first always starts the burst at
// word 0 (start_word = 0), so the CPU's requested ("critical") word
// arrives whenever the fixed sequential burst happens to reach it. A
// memory system WITH critical-word-first instead starts the burst AT the
// requested word (start_word = target_word), so it always arrives first.
// This function models one burst; main.cpp calls it twice per scenario
// (once each way) to compare the two policies.
long time_to_word(int line_words, int start_word, int target_word,
                   int base_latency, int cycles_per_word);
