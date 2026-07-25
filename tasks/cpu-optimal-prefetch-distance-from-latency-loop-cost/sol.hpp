#pragma once

// ---------------------------------------------------------------------
// Return the smallest integer distance d (in loop iterations) such that
// issuing a prefetch d iterations ahead hides a memory load of latency
// `mem_latency` cycles behind `loop_body_cycles` cycles of work per
// iteration: the smallest d with d * loop_body_cycles >= mem_latency,
// i.e. ceil(mem_latency / loop_body_cycles). Both inputs are positive.
int prefetch_distance(int mem_latency, int loop_body_cycles);

// ---------------------------------------------------------------------
// Simulate a straight-line loop of `n` iterations against a fixed-cost
// timeline: iteration i executes (consumes its data) at cycle
// i * loop_body_cycles. The prefetch that supplies iteration i's data
// is issued `distance` iterations earlier -- at iteration
// max(i - distance, 0) -- and lands (data becomes available) exactly
// `mem_latency` cycles after it is issued, i.e. at cycle
// max(i - distance, 0) * loop_body_cycles + mem_latency.
//
// Iteration i STALLS if its data has not landed by the time it is
// consumed: landing_cycle > i * loop_body_cycles. Return the number of
// stalling iterations across i in [0, n).
int count_stalls(int n, int mem_latency, int loop_body_cycles, int distance);
