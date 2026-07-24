#pragma once

// Arithmetic intensity (FLOPs/byte) of a kernel that touches some subset of
// the fields of a struct, for two different memory layouts.
//
//   struct_bytes  -- sizeof(the struct), as measured by the REAL compiler for
//                     one of the concrete structs in main.cpp (padding and
//                     alignment already baked in -- this is not reimplemented
//                     here, it is read straight from a real sizeof()).
//   field_bytes   -- sizeof of each field, in declaration order; field_bytes[i]
//                     is the size of field i as measured by the real compiler.
//   num_fields    -- length of field_bytes.
//   reads         -- indices into field_bytes that the kernel reads.
//   num_reads     -- length of reads.
//   writes        -- indices into field_bytes that the kernel writes.
//   num_writes    -- length of writes.
//   flops         -- floating point operations performed, per element.
//   is_aos        -- true: Array-of-Structs layout. false: Struct-of-Arrays.
//
// AoS: touching ANY field for reading pulls in the WHOLE struct (struct_bytes)
//      once; touching ANY field for writing pushes out the WHOLE struct
//      (struct_bytes) once. A kernel that both reads and writes therefore
//      moves 2 * struct_bytes per element -- the read traffic and the write
//      traffic are separate, even though it's the same struct.
// SoA: each field lives in its own array, so only the touched columns move:
//      bytes = sum(field_bytes[i] for i in reads) + sum(field_bytes[i] for i in writes).
//      (A field that is both read AND written counts once for the read and
//      once for the write, exactly like AoS -- it is still two passes over
//      memory in general.)
//
// Return flops / bytes_moved. If bytes_moved is 0 (kernel touches nothing),
// return +infinity (use std::numeric_limits<double>::infinity()).
double arithmetic_intensity(int struct_bytes,
                             const int* field_bytes, int num_fields,
                             const int* reads, int num_reads,
                             const int* writes, int num_writes,
                             int flops, bool is_aos);
