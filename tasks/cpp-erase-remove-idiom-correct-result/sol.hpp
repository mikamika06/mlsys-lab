#pragma once
#include <vector>

struct Record {
    int key;
    double val;
};

// Removes every element of v whose `key` equals `target`, using the
// erase-remove idiom: std::remove_if(v.begin(), v.end(), pred) followed by
// v.erase(...). The RELATIVE ORDER of every retained element must be
// preserved exactly, and EVERY matching element must be removed -- including
// runs of two or more matches in a row.
void eraseByKey(std::vector<Record>& v, int target);
