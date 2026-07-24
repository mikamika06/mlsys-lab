import itertools
import re


def enumerate_accepted_strings(regex, vocab, max_len):
    accepted = set()
    for length in range(max_len + 1):
        for parts in itertools.product(vocab, repeat=length):
            candidate = "".join(parts)
            if re.fullmatch(regex, candidate):
                accepted.add(candidate)
    return sorted(accepted)
