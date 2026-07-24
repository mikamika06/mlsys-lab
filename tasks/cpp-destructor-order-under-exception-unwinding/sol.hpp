#pragma once

// One step of a sequential trace through nested C++ scopes.
enum StmtKind { BEGIN, END, CONSTRUCT, THROW };

struct Stmt {
    StmtKind kind;
    int id;     // object id, meaningful for CONSTRUCT
    int bytes;  // sizeof this object (a REAL sizeof(), computed by main.cpp
                // from an actual struct type -- see main.cpp), meaningful
                // for CONSTRUCT
};

// Replay stmts[0..n) as sequential C++ execution:
//   BEGIN      enter a new nested lexical scope
//   END        leave the current (innermost) scope NORMALLY: every object
//              constructed in that scope is destructed, in REVERSE order of
//              construction -- ordinary destruction, not unwinding
//   CONSTRUCT  construct an object with the given id/bytes; it becomes live
//              in the current (innermost) scope
//   THROW      throw an exception. Execution stops at this statement:
//              STACK UNWINDING destroys every object that is STILL LIVE
//              (i.e. was CONSTRUCTed but not yet destroyed by an earlier
//              END), across every currently open scope, in reverse order of
//              construction -- irrespective of which scope each one lives
//              in, since scopes were opened in chronological order too.
//
// If a THROW is reached: fill out_ids[0 .. *out_count) with the ids
// destructed DURING UNWINDING ONLY, in the order their destructors actually
// fire; set *out_count to how many; return the sum of their `bytes`.
// If no THROW is reached: set *out_count = 0 and return 0.
long run_trace(const Stmt* stmts, int n, int* out_ids, int* out_count);
