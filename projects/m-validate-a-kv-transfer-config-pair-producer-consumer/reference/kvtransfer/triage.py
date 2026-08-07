def diagnose_stuck_handshake(producer_logs: list[dict], consumer_logs: list[dict]) -> dict:
    """Diagnose stuck handshake from producer and consumer log entries."""
    prod_events = {e["event"]: e for e in producer_logs}
    cons_events = {e["event"]: e for e in consumer_logs}

    if "INIT" not in prod_events:
        return {"reason": "PRODUCER_INIT_MISSING", "stuck": True}
    if "INIT" not in cons_events:
        return {"reason": "CONSUMER_INIT_MISSING", "stuck": True}

    prod_init = prod_events["INIT"]
    cons_init = cons_events["INIT"]

    if prod_init.get("session_id") != cons_init.get("session_id"):
        return {"reason": "SESSION_ID_MISMATCH", "stuck": True}

    if "MAGIC_ACK" in prod_events and prod_events["MAGIC_ACK"].get("status") == "REJECTED":
        return {"reason": "MAGIC_HEADER_REJECTED", "stuck": True}
    if "MAGIC_ACK" in cons_events and cons_events["MAGIC_ACK"].get("status") == "REJECTED":
        return {"reason": "MAGIC_HEADER_REJECTED", "stuck": True}

    if "PARAM_SYNC" in prod_events and "PARAM_SYNC" in cons_events:
        if prod_events["PARAM_SYNC"].get("mem_key") != cons_events["PARAM_SYNC"].get("mem_key"):
            return {"reason": "MEM_KEY_MISMATCH", "stuck": True}

    if "CONNECT" in prod_events and prod_events["CONNECT"].get("status") == "TIMEOUT":
        return {"reason": "CONNECTION_TIMEOUT", "stuck": True}
    if "CONNECT" in cons_events and cons_events["CONNECT"].get("status") == "TIMEOUT":
        return {"reason": "CONNECTION_TIMEOUT", "stuck": True}

    if "READY" in prod_events and "READY" in cons_events:
        return {"reason": "NONE", "stuck": False}

    return {"reason": "UNKNOWN_HANDSHAKE_HALT", "stuck": True}
