def diagnose_startup(init_payload: dict) -> str:
    """Diagnoses one of the 6 vLLM TP + EAGLE-3 startup failure outcomes."""
    raise NotImplementedError


def diagnose_all_outcomes(payloads: list[dict]) -> list[str]:
    """Diagnoses a sequence of startup init payloads."""
    raise NotImplementedError
