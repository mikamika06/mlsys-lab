def parse_host_string(host_str):
    if not host_str:
        return ("127.0.0.1", 11434)
    s = host_str.strip()
    if "://" in s:
        s = s.split("://", 1)[1]
    if "/" in s:
        s = s.split("/", 1)[0]

    if ":" in s:
        parts = s.rsplit(":", 1)
        host = parts[0] or "127.0.0.1"
        try:
            port = int(parts[1])
        except ValueError:
            port = 11434
        return (host, port)
    return (s, 11434)


def reconcile_host_binding(ollama_host_env, active_sockets):
    """Compare configured OLLAMA_HOST against active bound daemon sockets."""
    target_host, target_port = parse_host_string(ollama_host_env)

    matches = []
    for sock in active_sockets:
        shost = sock.get("host", "")
        sport = sock.get("port", 0)

        host_match = (target_host == shost or shost == "0.0.0.0" or target_host == "localhost" and shost == "127.0.0.1")
        port_match = (target_port == sport)

        if host_match and port_match:
            matches.append(sock)

    return {
        "configured_host": target_host,
        "configured_port": target_port,
        "is_reconciled": len(matches) > 0,
        "matched_sockets": matches
    }
