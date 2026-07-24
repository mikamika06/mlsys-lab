#pragma once
#include <string>

// Return the storage-duration label for each of the 20 variable
// declarations in task.md, in order, into out[0..20). Each label must be
// exactly one of: "automatic", "static", "thread", "dynamic".
void name_storage_durations(std::string out[20]);
