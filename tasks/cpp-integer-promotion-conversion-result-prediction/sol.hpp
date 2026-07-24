#pragma once

// The 8 integer types this task covers, at their real sizes on this ABI:
// char/uchar = 1 byte, short/ushort = 2 bytes, int/uint = 4 bytes,
// long/ulong = 8 bytes (char is SIGNED on this compiler/platform).
enum class IntType { Char, UChar, Short, UShort, Int, UInt, Long, ULong };

struct ConvResult {
    long long value;   // the result, as a bit pattern: read it signed if
                        // isSigned, else reinterpret as unsigned long long
    int width;          // sizeof of the result type, in bytes (4 or 8 --
                          // promotion means it's never 1 or 2)
    bool isSigned;       // whether the result type is signed
};

// Reproduces C++'s integer promotion + usual arithmetic conversions for a
// binary '+'/'-'/'*' between an operand of type lhsType (value lhsVal) and
// an operand of type rhsType (value rhsVal):
//
//   1. Any operand narrower than int (char/uchar/short/ushort) is PROMOTED
//      to int first -- its whole range fits in int, on this ABI.
//   2. If the (now int-or-wider) operand types still differ:
//        - same signedness -> the higher-rank type wins (rank order:
//          int/uint < long/ulong)
//        - different signedness, same rank -> the unsigned type wins
//        - different signedness, unsigned operand's rank >= signed
//          operand's rank -> the unsigned type wins
//        - different signedness, signed operand's rank > unsigned
//          operand's rank AND can represent the unsigned type's whole
//          range (true for long vs uint on this ABI) -> the signed type
//          wins
//   3. The operation is evaluated in that common type, wrapping modulo
//      2^width (unsigned: plain modular wraparound; signed: two's
//      complement -- and every case exercised by main.cpp is chosen so the
//      *signed* path never actually overflows, only the unsigned one does).
ConvResult evalConversion(char op, IntType lhsType, long long lhsVal,
                           IntType rhsType, long long rhsVal);
