#pragma once

// A tiny heap-backed vector type with an INSTRUMENTED copy counter, used to
// measure how many real deep copies each factory pattern performs.
// All of Matrix's members are harness code (defined in main.cpp) — you do
// not implement them. `Matrix(n)` allocates n ints and fills
// data[i] = i * i.
struct Matrix {
    int n;
    int* data;

    Matrix();                                 // n=0, data=nullptr
    explicit Matrix(int n_);                  // allocates + fills data[i] = i*i
    Matrix(const Matrix& other);              // deep copy; increments g_copy_count
    Matrix& operator=(const Matrix& other);   // deep copy; increments g_copy_count
    Matrix(Matrix&& other) noexcept;          // steals the buffer; NOT counted
    Matrix& operator=(Matrix&& other) noexcept; // steals the buffer; NOT counted
    ~Matrix();
};

// Reset to 0 by the driver immediately before each factory call below.
extern int g_copy_count;

// Factory 1 — by-value return. Implement this as exactly
//     return Matrix(n);
// i.e. construct and return a prvalue directly. Under C++17 "guaranteed
// copy elision" that construction happens straight into the caller's
// storage: NO copy constructor and NO move constructor runs — this is
// mandated by the standard, not just an optimizer trick, so g_copy_count
// stays 0 regardless of build flags.
Matrix make_by_value(int n);

// Factory 2 — out-parameter. This is the "avoid the copy" idiom people
// reach for instead of returning by value. Implement it the way it is most
// commonly written in real code — build a local Matrix, then copy-assign
// it into `out`:
//     Matrix tmp(n);
//     out = tmp;
// `tmp` is a named lvalue, so `out = tmp` calls the COPY-assignment
// operator (not move), which is exactly the point of the exercise: this
// idiom does not avoid the copy the way intuition suggests.
void make_out_param(int n, Matrix& out);
