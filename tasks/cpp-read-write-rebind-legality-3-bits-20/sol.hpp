#pragma once

// For each of the 20 declarations documented in task.md, predict 3
// legality bits:
//   may_read         -- is dereferencing/using it to READ a value legal?
//   may_write_through -- is assigning THROUGH it (to what it points/refers
//                        to) legal?
//   may_rebind        -- can the declared NAME ITSELF be made to
//                        point/refer to a different object after
//                        initialization? (References can NEVER be
//                        rebound in C++, regardless of what they refer
//                        to -- this bit is always 0 for every reference
//                        type.)
//
// out[i*3 + 0] = may_read, out[i*3 + 1] = may_write_through,
// out[i*3 + 2] = may_rebind, for declaration i (0-indexed, 20 declarations
// -> 60 entries total). Each entry is 0 or 1.
void predict_legality(int out[60]);
