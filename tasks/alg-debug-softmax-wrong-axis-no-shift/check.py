import numpy as np

def grade(sol, fx) -> dict:
    np.random.seed(42)
    
    shapes_axes = [
        ((10, 20), -1),
        ((5, 5, 5), 1),
        ((100,), 0),
        ((2, 3, 4, 5), 2)
    ]
    
    kl_divs = []
    
    def ref_softmax(x, axis):
        x_max = np.max(x, axis=axis, keepdims=True)
        exps = np.exp(x - x_max)
        return exps / np.sum(exps, axis=axis, keepdims=True)
        
    for shape, axis in shapes_axes:
        # Create inputs with large magnitude to trigger overflow without shift
        x = np.random.randn(*shape) * 1000.0 + 500.0
        
        ref_out = ref_softmax(x, axis=axis)
        
        try:
            sol_out = sol.softmax(x, axis=axis)
        except Exception as e:
            return {"mean_kl": 999.0}
            
        if not isinstance(sol_out, np.ndarray):
            return {"mean_kl": 999.0}
            
        if sol_out.shape != ref_out.shape:
            return {"mean_kl": 999.0}
            
        if np.any(np.isnan(sol_out)) or np.any(np.isinf(sol_out)):
            return {"mean_kl": 999.0}
            
        eps = 1e-12
        ref_safe = np.clip(ref_out, eps, 1.0)
        sol_safe = np.clip(sol_out, eps, 1.0)
        
        kl = np.sum(ref_safe * np.log(ref_safe / sol_safe), axis=axis)
        kl_divs.append(np.mean(kl))
        
    return {"mean_kl": float(np.mean(kl_divs))}
