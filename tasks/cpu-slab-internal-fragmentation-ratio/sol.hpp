#pragma once

// A slab allocator only hands out blocks from a fixed set of size
// classes (e.g. 16, 32, 64, ... bytes). A request for `r` bytes is
// rounded UP to the smallest size class >= r; the difference between
// what was allocated and what was requested is INTERNAL fragmentation.
//
// Given `num_classes` size classes (size_classes[0..num_classes), sorted
// ascending) and `n` requested sizes (requests[0..n), each
// <= size_classes[num_classes-1]), compute for each request i the
// smallest size class >= requests[i] and its ratio
// allocated / requests[i] (>= 1.0; 1.0 means no waste). Return the
// AVERAGE of these n ratios.
double slab_fragmentation_ratio(const int* size_classes, int num_classes, const int* requests, int n);
