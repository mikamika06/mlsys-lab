#pragma once
// Predict the final reference count of a CPython-style object (starting
// refcount 1, exactly like a freshly created PyObject) after applying each
// of the 12 fixed operation sequences in task.md. Each op in a sequence is
// one of "New", "Incref", "Decref", "Borrow" -- see task.md for what each
// one does to the count.
//
// Write your 12 predictions into out[0..12) (out[i] is sequence i+1).
void predict_refcounts(int out[12]);
