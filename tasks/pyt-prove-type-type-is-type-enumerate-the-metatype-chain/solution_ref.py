def metatype_chain(x):
    chain = []
    current = type(x)
    while True:
        chain.append(current.__name__)
        if current.__name__ == "type":
            break
        current = type(current)
    return chain
