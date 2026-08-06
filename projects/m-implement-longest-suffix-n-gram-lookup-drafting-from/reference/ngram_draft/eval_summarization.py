from ngram_draft.lookup import find_longest_suffix_match


def evaluate_summarization(doc_tokens, summary_prefix, draft_len, max_ngram_size):
    tokens = list(doc_tokens) + list(summary_prefix)
    steps = 0
    accepted_total = 0
    drafted_total = 0
    ground_truth = list(doc_tokens)
    idx = 0

    while idx < 50 and len(tokens) < 300:
        draft = find_longest_suffix_match(tokens, max_ngram_size, draft_len)
        if not draft:
            next_tok = ground_truth[idx % len(ground_truth)]
            tokens.append(next_tok)
            steps += 1
            idx += 1
            continue

        drafted_total += len(draft)
        accepted_in_step = 0
        for tok in draft:
            expected = ground_truth[idx % len(ground_truth)]
            if tok == expected:
                accepted_in_step += 1
                idx += 1
            else:
                break

        accepted_total += accepted_in_step
        next_tok = ground_truth[idx % len(ground_truth)]
        tokens.extend(draft[:accepted_in_step])
        tokens.append(next_tok)
        idx += 1
        steps += 1

    baseline_steps = idx
    speedup = float(baseline_steps) / float(steps) if steps > 0 else 1.0
    acc_rate = float(accepted_total) / float(drafted_total) if drafted_total > 0 else 0.0
    return {"speedup": speedup, "acceptance_rate": acc_rate, "tokens_generated": idx}
