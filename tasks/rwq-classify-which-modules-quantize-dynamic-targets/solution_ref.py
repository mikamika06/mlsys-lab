def classify_quantizable(names):
    quantizable = {"Linear", "LSTM"}
    return [1 if n in quantizable else 0 for n in names]
