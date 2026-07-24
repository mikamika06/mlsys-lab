#pragma once
#include <cstddef>

// A tiny instrumented heap so the harness can *see* a double free instead of
// crashing on it. Every allocation gets a unique id; freeing an id twice, or
// freeing an id that was never handed out, is recorded rather than fatal.
namespace heap {
    int  alloc(int value);      // returns a fresh id > 0
    void release(int id);       // id == 0 is a no-op, exactly like `delete nullptr`
    int  read(int id);          // value stored behind an id (0 if not live)
    int  live();                // ids currently allocated
    int  doubleFrees();         // ids released while not live
    void reset();
}

// Buffer owns one heap id. It must behave like a correct RAII type:
//   - copy   -> a NEW allocation holding the same value (deep copy)
//   - move   -> steals the id and leaves the source owning NOTHING
//   - assign -> releases whatever it already owned first, and self-assignment
//               must not destroy the object
// The bug this task is about: a move constructor that copies the id but forgets
// to clear the source, so both objects release the same id in their destructor.
class Buffer {
public:
    explicit Buffer(int value);
    ~Buffer();

    Buffer(const Buffer& other);              // deep copy
    Buffer(Buffer&& other) noexcept;          // steal, and null the source
    Buffer& operator=(const Buffer& other);   // deep copy, self-assign safe
    Buffer& operator=(Buffer&& other) noexcept;

    int value() const;                        // 0 when this Buffer owns nothing
    int id() const { return id_; }            // 0 == owns nothing

private:
    int id_;
};
