#include "sol.hpp"

// Correct move-only RAII handle with a custom deleter.

GpuHandle::GpuHandle(std::uint64_t id) : id_(id) {}

GpuHandle::~GpuHandle() {
    gpu_release(id_);            // releasing id 0 is a no-op, so empty is safe
}

GpuHandle::GpuHandle(GpuHandle&& other) noexcept : id_(other.id_) {
    other.id_ = 0;              // source no longer owns anything
}

GpuHandle& GpuHandle::operator=(GpuHandle&& other) noexcept {
    if (this != &other) {
        gpu_release(id_);       // free the resource we currently own
        id_ = other.id_;        // steal the source's resource
        other.id_ = 0;          // source becomes empty
    }
    return *this;
}

std::uint64_t GpuHandle::get() const {
    return id_;
}

void GpuHandle::reset() {
    gpu_release(id_);           // free now
    id_ = 0;                    // become empty (destructor won't double-free)
}
