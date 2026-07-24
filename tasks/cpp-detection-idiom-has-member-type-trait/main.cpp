#include <cstdio>
#include "sol.hpp"

// FIXED driver: instantiates all 12 detectors and prints each as 0/1.

int main() {
    bool r[12] = {
        detect_DProbe1(),  detect_DProbe2(),  detect_DProbe3(),  detect_DProbe4(),
        detect_DProbe5(),  detect_DProbe6(),  detect_DProbe7(),  detect_DProbe8(),
        detect_DProbe9(),  detect_DProbe10(), detect_DProbe11(), detect_DProbe12(),
    };
    for (int i = 0; i < 12; ++i) {
        printf("%d ", r[i] ? 1 : 0);
    }
    printf("\n");
    return 0;
}
