from ngram.index import PromptNgramIndex
from ngram.policy import select_candidates, should_disable
from ngram.engine import NgramSpeculativeEngine

def create_oracle_index(prompt, n=4):
    return PromptNgramIndex(prompt, n=n)

def oracle_select(index, tokens, k=4):
    return select_candidates(index, tokens, k=k)

def oracle_disable(history, threshold=0.1):
    return should_disable(history, threshold=threshold)
