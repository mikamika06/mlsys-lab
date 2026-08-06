from ngram_draft.lookup import find_longest_suffix_match


def evaluate_copy(text_tokens, max_ngram_size, draft_len):
    tokens = list(text_tokens[:20])
    ground_truth = list(text_tokens[20:])
    pos = 0
    accepted = 0
    drafted = 0

    while pos < len(ground_truth):
        draft = find_longest_suffix_match(tokens, max_ngram_size, draft_len)
        if not draft:
            tokens.append(ground_truth[pos])
            pos += 1
            continue

        drafted += len(draft)
        matched = 0
        for d in draft:
            if pos < len(ground_truth) and d == ground_truth[pos]:
                matched += 1
                pos += 1
            else:
                break
        accepted += matched
        tokens.extend(draft[:matched])
        if pos < len(ground_truth):
            tokens.append(ground_truth[pos])
            pos += 1

    acc_rate = float(accepted) / float(drafted) if drafted > 0 else 0.0
    return {"acceptance_rate": acc_rate, "total_drafted": drafted, "total_accepted": accepted}
