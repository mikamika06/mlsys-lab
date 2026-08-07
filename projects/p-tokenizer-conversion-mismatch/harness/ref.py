import random


def generate_corpus(seed=42):
    rng = random.Random(seed)
    words = ["hello", "world", "mlsys", "tokenizer", "GGUF", "llama", "test", "case", "byte", "utf8", "   ", "\n", "\t", "123", "!@#"]
    corpus = []
    for _ in range(100):
        length = rng.randint(5, 20)
        s = "".join(rng.choice(words) + (" " if rng.random() > 0.3 else "") for _ in range(length))
        corpus.append(s)
    return corpus


def oracle_tokenize(text):
    return [ord(c) for c in text]


def oracle_detokenize(tokens):
    return "".join(chr(t) for t in tokens)
