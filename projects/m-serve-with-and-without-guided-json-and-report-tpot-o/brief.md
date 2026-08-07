When deploying Large Language Models via serving frameworks like vLLM, enforcing structured outputs such as `guided_json` guarantees that generated tokens adhere strictly to a predefined JSON schema. However, applying grammar or schema constraints on the fly restricts the allowed token sampling space at each generation step, altering token generation dynamics.

Engineers need a systematic way to measure, benchmark, and report the Time Per Output Token (TPOT) overhead introduced by enabling `guided_json` compared to unconstrained text generation. Without this evaluation, service level objectives for generation latency can be severely compromised under structured decoding workloads.

Your task is to implement a serving performance analysis pipeline that simulates execution traces with and without `guided_json`, computes the resulting TPOT overhead ratio, and packages a regression test suite to safeguard the measurement logic.
