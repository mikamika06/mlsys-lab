#pragma once

// ---------------------------------------------------------------------------
// Deterministic memory-level-parallelism (MLP) model, standing in for real
// hardware latency (not reproducible/timeable, so never measured directly).
//
// A memory subsystem can have several requests "in flight" at once, up to
// some finite number of outstanding-request trackers (MSHRs). Model that
// as WAVES: every access you schedule gets a wave_id; all accesses sharing
// a wave_id are considered issued concurrently (they overlap, costing ONE
// latency unit together), while accesses in different waves are serial
// (each costs its own latency unit). The workload below has TWO kinds of
// access:
//
//   - NUM_CHASE_STEPS genuinely data-dependent pointer-chase steps
//     (access_id 0..NUM_CHASE_STEPS-1): step k+1's target address is only
//     known after step k's VALUE has been loaded, so they are
//     unavoidably serial.
//   - NUM_EMBED_LOOKUPS genuinely independent embedding-table lookups
//     (access_id NUM_CHASE_STEPS..TOTAL_ACCESSES-1): every address is
//     known upfront from input indices, not from another lookup's
//     result, so they can freely share waves with anything (subject only
//     to the width limit).
// ---------------------------------------------------------------------------
constexpr int NUM_CHASE_STEPS = 4;
constexpr int NUM_EMBED_LOOKUPS = 24;
constexpr int MAX_WAVE_WIDTH = 8;   // at most this many accesses per wave
constexpr int TOTAL_ACCESSES = NUM_CHASE_STEPS + NUM_EMBED_LOOKUPS;

// PROVIDED (defined in main.cpp).
void schedule_access(int access_id, int wave_id);
long modeled_cycles();

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS.
//
// Call schedule_access(access_id, wave_id) EXACTLY ONCE for every access_id
// in [0, TOTAL_ACCESSES), assigning each a wave_id (any non-negative
// integers, however many distinct waves you like), subject to:
//
//   1. For every k in [0, NUM_CHASE_STEPS - 1): the wave_id of chase step
//      k+1 must be STRICTLY GREATER than the wave_id of chase step k (it
//      depends on step k's loaded value, so it cannot be issued in the
//      same or an earlier wave).
//   2. No more than MAX_WAVE_WIDTH accesses (of any kind) may share the
//      same wave_id.
//   3. Embedding lookups have no ordering constraint of their own -- pack
//      them into whichever waves (including the chase-step waves) have
//      spare capacity, to MINIMIZE the total number of distinct wave ids
//      used across the whole schedule.
//
// modeled_cycles() reports that total distinct-wave count (or a huge
// sentinel if your schedule violates rule 1 or 2). Minimizing it is the
// whole exercise: the naive schedule that gives every access its own wave
// is always valid but never optimal.
// ---------------------------------------------------------------------------
void schedule_embedding_workload();
