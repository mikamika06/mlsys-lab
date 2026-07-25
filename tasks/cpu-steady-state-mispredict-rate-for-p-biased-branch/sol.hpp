#pragma once

// LEARNER IMPLEMENTS.
//
// A 2-bit saturating counter branch predictor has 4 states -- Strongly
// Not-taken=0, Weakly Not-taken=1, Weakly Taken=2, Strongly Taken=3 --
// and predicts TAKEN whenever state >= 2, else NOT-TAKEN. After the
// real outcome is known, the state saturates up by one on an actual
// TAKEN (capped at 3) or down by one on an actual NOT-TAKEN (capped at
// 0).
//
// For a branch that is independently TAKEN with fixed probability `p`
// on every single execution (a memoryless i.i.d. Bernoulli(p) stream --
// there is no temporal PATTERN for the predictor to lock onto, only the
// raw bias), derive the counter's STEADY-STATE mispredict rate: the
// long-run fraction of predictions that turn out wrong, once the
// counter's state distribution has settled into equilibrium.
//
// This has a closed form. Model the 4 states as a birth-death Markov
// chain (state i -> i+1 with probability p, state i -> i-1 with
// probability q = 1-p, and the two boundary states self-loop on the
// move that would go out of range). Solve for the chain's stationary
// distribution pi via detailed balance (pi_i * p = pi_{i+1} * q for
// each adjacent pair), then the steady-state mispredict rate is
//
//   p * (pi_0 + pi_1) + (1 - p) * (pi_2 + pi_3)
//
// -- mispredicting a TAKEN branch (probability p) while parked in one
// of the two NOT-TAKEN-predicting states, plus mispredicting a
// NOT-TAKEN branch (probability q) while parked in one of the two
// TAKEN-predicting states. Working the algebra through collapses this
// to a short closed-form expression in p and q alone.
double steady_state_mispredict_rate(double p);
