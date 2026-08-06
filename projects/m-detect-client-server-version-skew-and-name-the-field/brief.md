# Local Runner Architecture: Client/Server Skew and Daemon Binding Diagnostics

Users on the local team are reporting flaky behavior when interacting with local LLM runners (specifically Ollama instances). In several developer environments, CLI calls silently behave unpredictably or fail to connect despite the daemon process running.

Three main symptoms have been observed in local logs:
1. Some CLI commands succeed partially but silently misuse feature flags, suspecting version mismatches between the client binary and the running daemon process.
2. CLI instances fail to reach local daemons, or connect to wrong instances, due to host/port discrepancies between environment settings like `OLLAMA_HOST` and actual bound network sockets.
3. Cold-start requests immediately after launching `ollama serve` experience severe initial latency spikes, but engineers lack automated instrumentation to isolate fixed server startup overhead from actual inference runtime.

Your task is to build a diagnostic tool module in `runnerdiag/` that:
1. Inspects API responses to detect version skew between client and server, identifying the exact JSON payload field proving the discrepancy.
2. Probes active local daemon host/port bindings and reconciles them against `OLLAMA_HOST` configurations.
3. Measures first-request cold latency against warm requests to isolate cold-start overheads accurately.
4. Provides regression test coverage in `tests/test_regression.py` that catches undetected version skew edge cases.
