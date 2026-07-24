#include "sol.hpp"

void transactional_update(Record* rec, const Op* ops, int numOps, int throwAt) {
    char type = rec->type;
    double score = rec->score;
    int flags = rec->flags;
    long id = rec->id;

    for (int i = 0; i < numOps; i++) {
        if (i == throwAt) throw TxnAbort();
        switch (ops[i].field) {
            case Field::Type:  type  = (char)ops[i].value; break;
            case Field::Score: score = ops[i].value; break;
            case Field::Flags: flags = (int)ops[i].value; break;
            case Field::Id:    id    = (long)ops[i].value; break;
        }
    }

    // single non-throwing commit -- everything or nothing reaches *rec
    rec->type = type;
    rec->score = score;
    rec->flags = flags;
    rec->id = id;
}
