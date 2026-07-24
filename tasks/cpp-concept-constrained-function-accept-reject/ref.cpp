#include "sol.hpp"
#include <concepts>

template<class T>
concept Acceptable = requires(T x) {
    { x + x } -> std::same_as<T>;
};

void classify_accepts(int out[12]) {
    out[0]  = Acceptable<int> ? 1 : 0;
    out[1]  = Acceptable<double> ? 1 : 0;
    out[2]  = Acceptable<float> ? 1 : 0;
    out[3]  = Acceptable<bool> ? 1 : 0;
    out[4]  = Acceptable<long> ? 1 : 0;
    out[5]  = Acceptable<char> ? 1 : 0;
    out[6]  = Acceptable<SelfAdd> ? 1 : 0;
    out[7]  = Acceptable<DifferentReturn> ? 1 : 0;
    out[8]  = Acceptable<MissingAdd> ? 1 : 0;
    out[9]  = Acceptable<AmbiguousAdd> ? 1 : 0;
    out[10] = Acceptable<DeletedAdd> ? 1 : 0;
    out[11] = Acceptable<MixedOnly> ? 1 : 0;
}
