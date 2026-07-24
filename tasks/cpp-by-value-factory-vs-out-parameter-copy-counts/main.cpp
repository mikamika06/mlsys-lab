#include <cstdio>
#include "sol.hpp"

int g_copy_count = 0;

Matrix::Matrix() : n(0), data(nullptr) {}

Matrix::Matrix(int n_) : n(n_), data(new int[n_]) {
    for (int i = 0; i < n_; i++) data[i] = i * i;
}

Matrix::Matrix(const Matrix& other) : n(other.n), data(new int[other.n]) {
    for (int i = 0; i < n; i++) data[i] = other.data[i];
    g_copy_count++;
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this == &other) return *this;
    int* fresh = new int[other.n];
    for (int i = 0; i < other.n; i++) fresh[i] = other.data[i];
    delete[] data;
    data = fresh;
    n = other.n;
    g_copy_count++;
    return *this;
}

Matrix::Matrix(Matrix&& other) noexcept : n(other.n), data(other.data) {
    other.n = 0;
    other.data = nullptr;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this == &other) return *this;
    delete[] data;
    data = other.data;
    n = other.n;
    other.n = 0;
    other.data = nullptr;
    return *this;
}

Matrix::~Matrix() { delete[] data; }

static void print_matrix(const Matrix& m) {
    printf("[");
    for (int i = 0; i < m.n; i++) printf("%d%s", m.data[i], (i + 1 < m.n) ? "," : "");
    printf("]");
}

int main() {
    const int NF = 3;
    int fixtures[NF] = {1, 3, 5};

    for (int f = 0; f < NF; f++) {
        int n = fixtures[f];

        g_copy_count = 0;
        Matrix bv = make_by_value(n);
        int bv_copies = g_copy_count;

        g_copy_count = 0;
        Matrix op;
        make_out_param(n, op);
        int op_copies = g_copy_count;

        printf("n=%d by_value=", n);
        print_matrix(bv);
        printf(" copies=%d out_param=", bv_copies);
        print_matrix(op);
        printf(" copies=%d\n", op_copies);
    }
    return 0;
}
