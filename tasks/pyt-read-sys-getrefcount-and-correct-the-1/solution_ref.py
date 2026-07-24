import sys


def true_refcount(x):
    return sys.getrefcount(x) - 2
