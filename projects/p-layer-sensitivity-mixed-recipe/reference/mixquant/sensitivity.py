from mixquant.quantizer import run_forward, evaluate_model

def measure_sensitivity(weights, inputs, candidate_bits=(2, 4, 8)):
    ref_out = run_forward(weights, [32] * len(weights), inputs)
    sens = {}
    num_layers = len(weights)
    for i in range(num_layers):
        sens[i] = {}
        for b in candidate_bits:
            recipe = [32] * num_layers
            recipe[i] = b
            mse = evaluate_model(weights, recipe, inputs, ref_output=ref_out)
            sens[i][b] = float(mse)
    return sens
