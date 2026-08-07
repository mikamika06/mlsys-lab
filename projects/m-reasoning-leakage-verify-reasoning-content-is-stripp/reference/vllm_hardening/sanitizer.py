def sanitize_response(response, is_untrusted=True):
    if not is_untrusted:
        return response
    if isinstance(response, dict):
        new_resp = {}
        for k, v in response.items():
            if k == "reasoning_content":
                continue
            if k == "choices" and isinstance(v, list):
                new_choices = []
                for c in v:
                    if isinstance(c, dict):
                        new_c = {}
                        for ck, cv in c.items():
                            if ck == "reasoning_content":
                                continue
                            if ck in ("message", "delta") and isinstance(cv, dict):
                                new_msg = {}
                                for mk, mv in cv.items():
                                    if mk == "reasoning_content":
                                        continue
                                    new_msg[mk] = sanitize_response(mv, is_untrusted)
                                new_c[ck] = new_msg
                            else:
                                new_c[ck] = sanitize_response(cv, is_untrusted)
                        new_choices.append(new_c)
                    else:
                        new_choices.append(sanitize_response(c, is_untrusted))
                new_resp[k] = new_choices
            else:
                new_resp[k] = sanitize_response(v, is_untrusted)
        return new_resp
    elif isinstance(response, list):
        return [sanitize_response(item, is_untrusted) for item in response]
    else:
        return response


def sanitize_stream_chunk(chunk, is_untrusted=True):
    return sanitize_response(chunk, is_untrusted)
