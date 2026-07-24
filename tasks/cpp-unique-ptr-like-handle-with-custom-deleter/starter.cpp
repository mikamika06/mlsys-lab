#include "sol.hpp"

// TODO: implement a MOVE-ONLY RAII handle whose destructor releases the owned
// GPU id EXACTLY ONCE via the custom deleter gpu_release(). A moved-from handle
// must own nothing, and move-assignment must free the currently-owned id before
// stealing the source's id. reset() must free now and leave the handle empty.
//
// The bodies below compile and link but do not release anything yet, so every
// scenario in the driver reports 0 releases and the gate fails.

GpuHandle::GpuHandle(std::uint64_t id) : id_(id) {}

GpuHandle::~GpuHandle() {
    // your code here
}

GpuHandle::GpuHandle(GpuHandle&& other) noexcept : id_(other.id_) {
    // your code here
}

GpuHandle& GpuHandle::operator=(GpuHandle&& other) noexcept {
    // your code here
    (void)other;
    return *this;
}

std::uint64_t GpuHandle::get() const {
    return id_;
}

void GpuHandle::reset() {
    // your code here
}
