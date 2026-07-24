#include <cstdio>
#include "sol.hpp"

char g_log[64];
int g_log_len = 0;

void log_event(char c) {
    if (g_log_len < 63) g_log[g_log_len++] = c;
}

Probe::Probe(int id_) : id(id_) { log_event('C'); }
Probe::Probe(const Probe& other) : id(other.id) { log_event('Y'); }
Probe::Probe(Probe&& other) noexcept : id(other.id) { log_event('M'); }
Probe::~Probe() { log_event('D'); }

// FIXED driver: run reproduce_sequence(), then print whatever ended up in
// the log.
int main() {
    reproduce_sequence();
    g_log[g_log_len] = '\0';
    printf("log=%s len=%d\n", g_log, g_log_len);
    return 0;
}
