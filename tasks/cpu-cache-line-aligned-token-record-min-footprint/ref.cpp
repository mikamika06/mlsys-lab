#include "sol.hpp"
#include <cstdint>
#include <cstddef>

// Hot fields grouped first (no gaps between them: uint32_t, uint32_t,
// uint8_t all land back-to-back), so all 3 sit in the first cache line of
// every record. Cold fields follow with the compiler's own natural
// alignment padding -- nothing extra added.
struct TokenRecord {
    uint32_t id;
    uint32_t count;
    uint8_t flags;
    char name[24];
    uint64_t ts;
};

size_t record_size() { return sizeof(TokenRecord); }
size_t offset_id() { return offsetof(TokenRecord, id); }
size_t offset_count() { return offsetof(TokenRecord, count); }
size_t offset_flags() { return offsetof(TokenRecord, flags); }
size_t offset_name() { return offsetof(TokenRecord, name); }
size_t offset_ts() { return offsetof(TokenRecord, ts); }
