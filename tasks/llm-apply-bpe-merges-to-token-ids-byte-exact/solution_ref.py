def apply_bpe_merges(text, merges, vocab):
    tokens = list(text)
    merge_map = {(a,b): a+b for a,b in merges}
    while True:
        new_tokens=[]
        i=0
        changed=False
        while i < len(tokens):
            if i+1 < len(tokens) and (tokens[i], tokens[i+1]) in merge_map:
                merged_token = merge_map[(tokens[i], tokens[i+1])]
                new_tokens.append(merged_token)
                i+=2
                changed=True
            else:
                new_tokens.append(tokens[i])
                i+=1
        if not changed:
            break
        tokens=new_tokens
    return [vocab[t] for t in tokens]
