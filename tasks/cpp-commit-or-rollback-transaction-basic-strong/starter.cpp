#include "sol.hpp"

// BUG: writes straight into *rec as each op is applied. This only gives the
// BASIC exception guarantee (no corruption, no leak) -- if the sequence
// throws partway through, every op applied before the throw is already
// visible in *rec. Fix it to buffer into locals and commit once at the end.
void transactional_update(Record* rec, const Op* ops, int numOps, int throwAt) {
    for (int i = 0; i < numOps; i++) {
        if (i == throwAt) throw TxnAbort();
        switch (ops[i].field) {
            case Field::Type:  rec->type  = (char)ops[i].value; break;
            case Field::Score: rec->score = ops[i].value; break;
            case Field::Flags: rec->flags = (int)ops[i].value; break;
            case Field::Id:    rec->id    = (long)ops[i].value; break;
        }
    }
}
