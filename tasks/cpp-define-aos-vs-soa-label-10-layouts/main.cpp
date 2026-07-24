#include <cstdio>
#include "sol.hpp"

int main() {
    // 10 fixed struct layouts, each a small array of fields.
    static const Field L0[] = {{"float", false}, {"float", false}, {"float", false}};                 // Vec3 {x,y,z}
    static const Field L1[] = {{"float", true}, {"float", true}, {"float", true}};                     // 3 parallel arrays
    static const Field L2[] = {{"float", false}, {"float", false}, {"float", false}, {"int", false}};  // {vx,vy,vz,mass}
    static const Field L3[] = {{"float", false}, {"float", false}, {"float", false}, {"float", false}}; // Color {r,g,b,a}
    static const Field L4[] = {{"float", true}, {"float", true}};                                      // positions[], velocities[]
    static const Field L5[] = {{"int", false}, {"float", false}, {"float", false}, {"float", false}};  // {id,score,x,y}
    static const Field L6[] = {{"int", false}, {"int", false}, {"uint8_t", true}};                     // {w,h,pixels[]}
    static const Field L7[] = {{"float", true}};                                                        // 4x4 matrix m[16]
    static const Field L8[] = {{"float", false}, {"float", false}, {"float", false}, {"int", false}};  // {x0,x1,x2,count}
    static const Field L9[] = {{"int", false}, {"int", false}, {"int", false}, {"float", false}, {"float", false}, {"float", false}}; // {v0,v1,v2,nx,ny,nz}

    struct { const Field* fields; int n; } layouts[10] = {
        {L0, 3}, {L1, 3}, {L2, 4}, {L3, 4}, {L4, 2},
        {L5, 4}, {L6, 3}, {L7, 1}, {L8, 4}, {L9, 6},
    };

    for (int i = 0; i < 10; i++) {
        int label = classify_layout(layouts[i].fields, layouts[i].n);
        printf("%d ", label);
    }
    printf("\n");
    return 0;
}
