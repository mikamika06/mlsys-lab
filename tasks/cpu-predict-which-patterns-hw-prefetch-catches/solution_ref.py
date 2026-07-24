def classify_prefetch() -> list:
    """
    Return True/False for each of 5 access patterns:
    True = caught by hw prefetcher, False = not caught.

    Pattern 0: sequential (stride 4B)    -> True
    Pattern 1: fixed stride 16B          -> True
    Pattern 2: random                    -> False
    Pattern 3: pointer chase             -> False
    Pattern 4: large stride (4096B/page) -> False
    """
    return [True, True, False, False, False]
