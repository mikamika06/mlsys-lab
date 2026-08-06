def compare_strategies(abf_ppl: float, yarn_ppl: float) -> dict:
    diff = float(yarn_ppl) - float(abf_ppl)
    better = "abf" if abf_ppl < yarn_ppl else "yarn" if yarn_ppl < abf_ppl else "tie"
    return {"perplexity_difference": diff, "preferred": better}
