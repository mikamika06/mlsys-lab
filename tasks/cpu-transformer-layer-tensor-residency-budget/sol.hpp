#pragma once

// One transformer layer touches a fixed list of `num_tensors` tensors
// (weights and activations), tensor `i` being `tensor_bytes[i]` bytes and
// read `num_uses[i]` times over the layer's forward pass. The machine has
// `num_levels` cache levels, `cache_capacities[0..num_levels)` bytes each
// and STRICTLY INCREASING (`cache_capacities[0] < cache_capacities[1] <
// ...`) -- level 0 is the smallest/fastest, level `num_levels-1` the
// largest/slowest-of-the-cached-levels; beyond that is DRAM.
//
// Classify every tensor:
//   RESIDENT in the SMALLEST level L with tensor_bytes[i] <=
//   cache_capacities[L] -- it is loaded from DRAM ONCE (compulsory) and
//   every subsequent use is served from that cache level, so it costs
//   exactly tensor_bytes[i] bytes of DRAM traffic no matter how large
//   num_uses[i] is.
//
//   STREAMED (doesn't fit in even the LARGEST cache level, i.e.
//   tensor_bytes[i] > cache_capacities[num_levels-1]) -- every one of
//   its num_uses[i] reads has to come from DRAM again, costing
//   tensor_bytes[i] * num_uses[i] bytes of DRAM traffic.
//
// Write residency_out[i] = the level index a resident tensor fits in, or
// -1 for a streamed tensor. Return the layer's TOTAL modeled DRAM byte
// budget: the sum, over every tensor, of the DRAM traffic it costs under
// the rule above.
long classify_layer_residency(const long* tensor_bytes, const int* num_uses, int num_tensors,
                               const long* cache_capacities, int num_levels, int* residency_out);
