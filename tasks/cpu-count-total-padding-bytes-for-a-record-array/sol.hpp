#pragma once

// A record type is described as `num_fields` fields, declared in order,
// each with a byte size (`field_sizes[i]`) and a required alignment
// (`field_aligns[i]`, a power of two). The compiler lays fields out at
// increasing offsets, inserting padding before a field so its offset is
// a multiple of its own alignment (inter-field padding), then pads the
// whole record's size up to a multiple of the record's own alignment --
// the largest of its fields' alignments -- so that consecutive records
// packed back-to-back in an array all stay properly aligned (tail
// padding).
//
// Return the TOTAL padding bytes -- inter-field padding plus tail
// padding, summed -- across an array of `count` such records:
//   count * (padded_record_size - sum_of_field_sizes)
long total_padding_bytes(const int* field_sizes, const int* field_aligns, int num_fields, long count);
