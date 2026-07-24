#pragma once

// LP64-layout record: char, double, int, long.
struct Record {
    char type;
    double score;
    int flags;
    long id;
};

// Which field an Op targets.
enum class Field { Type, Score, Flags, Id };

// One update: a field selector and the new value (carried as a double;
// apply it with the correct cast for that field's real type).
struct Op {
    Field field;
    double value;
};

// Thrown by transactional_update to simulate a mid-transaction failure.
struct TxnAbort {};

// Applies ops[0..numOps) to *rec, one at a time. If `throwAt` equals the
// (0-based) index of an operation, throw TxnAbort() instead of applying that
// operation (or any later one). throwAt == -1 means never throw.
//
// This must provide the STRONG exception guarantee: on throw, *rec must be
// left byte-for-byte exactly as it was on entry -- not partially updated.
// Read the current fields into local copies, apply every op to the copies,
// and only write the copies back into *rec with a single non-throwing
// commit at the very end, once the whole sequence has succeeded.
void transactional_update(Record* rec, const Op* ops, int numOps, int throwAt);
