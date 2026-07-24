#include "sol.hpp"

#include <string>
#include <vector>

// Reference: same 12 sequences as main.cpp, predicted arithmetically --
// rc = 1 + #New + #Incref - #Decref (Borrow never changes the count).
void predict_refcounts(int out[12]) {
    static const std::vector<std::vector<std::string>> sequences = {
        {"New"},
        {"Incref", "Decref"},
        {"Incref", "Incref", "Decref"},
        {"Borrow", "Incref"},
        {"Incref", "Borrow", "Decref", "Decref"},
        {"New", "Incref", "Borrow", "Decref", "Decref"},
        {"New", "New"},
        {"Borrow", "Borrow"},
        {"Incref", "Incref", "Incref", "Decref", "Decref", "Decref"},
        {"Decref"},
        {"Incref", "Decref", "Incref", "Decref", "Incref"},
        {"New", "Borrow", "Incref", "Decref", "New", "Decref"},
    };
    for (int i = 0; i < 12; i++) {
        int rc = 1;
        for (const auto& op : sequences[i]) {
            if (op == "New" || op == "Incref") {
                rc++;
            } else if (op == "Decref") {
                rc--;
            }
        }
        out[i] = rc;
    }
}
