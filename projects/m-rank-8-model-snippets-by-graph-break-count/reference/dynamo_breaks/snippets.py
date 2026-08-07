import torch

def s_0_breaks(x):
    return x * 2

def s_1_break(x):
    y = x * 2
    print("break")
    return y + 1

def s_2_breaks(x):
    y = x * 2
    print("b1")
    y = y + 1
    print("b2")
    return y + 1

def s_3_breaks(x):
    y = x * 2
    print("b1")
    y = y + 1
    print("b2")
    y = y + 2
    print("b3")
    return y + 3

def s_4_breaks(x):
    y = x * 2
    print("b1")
    y = y + 1
    print("b2")
    y = y + 2
    print("b3")
    y = y + 3
    print("b4")
    return y + 4

def s_5_breaks(x):
    y = x * 2
    print("b1")
    y = y + 1
    print("b2")
    y = y + 2
    print("b3")
    y = y + 3
    print("b4")
    y = y + 4
    print("b5")
    return y + 5

def s_6_breaks(x):
    y = x * 2
    print("b1")
    y = y + 1
    print("b2")
    y = y + 2
    print("b3")
    y = y + 3
    print("b4")
    y = y + 4
    print("b5")
    y = y + 5
    print("b6")
    return y + 6

def s_7_breaks(x):
    y = x * 2
    for i in range(7):
        print("b")
        y = y + 1
    return y

def nested_if_fn(x):
    if x.sum().item() > 0:
        y = x * 2
        if y.mean().item() > 0:
            return y + 1
        return y - 1
    else:
        y = x * 3
        if y.mean().item() > 0:
            return y + 2
        return y - 2

SNIPPETS = {
    "s0": s_0_breaks,
    "s1": s_1_break,
    "s2": s_2_breaks,
    "s3": s_3_breaks,
    "s4": s_4_breaks,
    "s5": s_5_breaks,
    "s6": s_6_breaks,
    "s7": s_7_breaks,
}
