def compare_tps(ollama_tps, llama_cpp_tps):
    return {
        "ollama_tps": float(ollama_tps),
        "llama_cpp_tps": float(llama_cpp_tps),
        "ratio": float(ollama_tps / (llama_cpp_tps + 1e-9))
    }
