def check_version_skew(client_version, version_response_payload):
    """Detect client/server version skew and return status dict."""
    server_ver = version_response_payload.get("version", "")
    has_skew = (client_version != server_ver)
    return {
        "has_skew": has_skew,
        "client_version": client_version,
        "server_version": server_ver,
        "proving_field": "version"
    }


def identify_skew_field():
    """Return the name of the API field that identifies server version."""
    return "version"
