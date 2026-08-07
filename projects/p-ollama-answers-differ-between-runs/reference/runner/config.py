def merge_options(modelfile_opts, api_opts, req_opts):
    result = {}
    if modelfile_opts:
        result.update(modelfile_opts)
    if api_opts:
        result.update(api_opts)
    if req_opts:
        result.update(req_opts)
    return result
