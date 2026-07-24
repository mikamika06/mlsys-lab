#pragma once
#include <cstdint>

// Workload constants, DEFINED in main.cpp.
extern const int NUM_PARTICLES;
extern const int NUM_FIELDS;

// Hook declared here, DEFINED in main.cpp: records one memory access (as a
// BYTE address) against the deterministic direct-mapped cache model that
// main.cpp drives. Call it once per field value read, in the exact order
// the workload reads them.
void cacheTouch(int64_t byteAddr);

// Generates the access trace for a mixed physics-style workload reading
// AoSoA (Array-of-Structures-of-Arrays) data with tile width `tileWidth`.
//
// Layout: NUM_PARTICLES particles (see main.cpp), each with NUM_FIELDS
// float (4-byte) fields x, y, z, mass (field indices 0, 1, 2, 3), stored in
// AoSoA tiles of `tileWidth` particles: within tile t, field f's values for
// all `tileWidth` particles in that tile are stored contiguously
// (SoA-within-a-tile), and tiles are laid out back-to-back one after
// another (AoS-of-tiles). So the byte address of particle i's field f is:
//
//   tileIdx     = i / tileWidth
//   withinTile  = i % tileWidth
//   tileBytes   = tileWidth * NUM_FIELDS * 4        (bytes per whole tile)
//   fieldOffset = f * tileWidth * 4                 (bytes to field f's row)
//   addr        = tileIdx * tileBytes + fieldOffset + withinTile * 4
//
// The workload reads fields x, y, z (field indices 0, 1, 2 -- NOT mass, 3)
// for every particle i in order 0 .. NUM_PARTICLES-1, calling
// cacheTouch(addr) once per field read, three calls per particle, in field
// order x, then y, then z.
void generateAoSoATrace(int tileWidth);
