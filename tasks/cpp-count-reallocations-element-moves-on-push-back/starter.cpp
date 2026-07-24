#include "sol.hpp"

// TODO: simulate N push_backs of temporary Elements into a growable raw
// array following std::vector's reallocation/move policy. See sol.hpp for
// the exact contract. Every relocation must go through Element's real move
// constructor so g_move_count is incremented for real.
long simulate_vector_pushes(int N, int initial_capacity, int growth_factor) {
    (void)N;
    (void)initial_capacity;
    (void)growth_factor;
    return 0;
}
