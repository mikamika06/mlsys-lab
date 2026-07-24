#include "sol.hpp"

// An edit is ABI-breaking iff the in-memory OBJECT layout changes: total size,
// alignment, presence of a vptr, or the byte offset of any field that exists in
// both versions. If all of those are identical, existing compiled code keeps
// reading the same bytes from the same places -> ABI-compatible.
int abi_breaks(const Layout& old_v, const Layout& new_v) {
    if (old_v.size  != new_v.size)  return 1;
    if (old_v.align != new_v.align) return 1;
    if (old_v.vptr  != new_v.vptr)  return 1;
    int m = old_v.nfields < new_v.nfields ? old_v.nfields : new_v.nfields;
    for (int i = 0; i < m; i++)
        if (old_v.off[i] != new_v.off[i]) return 1;
    return 0;
}
