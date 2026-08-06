import ref


def check(workdir):
    from gotmpl import render

    plain = ref.pick("ollama", templates=["gpt-oss"], with_tools=False)
    tools = ref.pick("ollama", templates=["gpt-oss"], with_tools=True)
    return {"harmony_plain": ref.score(render, plain)[0],
            "harmony_tools": ref.score(render, tools)[0],
            "cases_total": float(len(plain) + len(tools))}
