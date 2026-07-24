def count_fib_frames(n: int) -> int:
    import sys

    count = 0

    def fib(n):
        nonlocal count
        if n <= 1:
            return n
        return fib(n - 1) + fib(n - 2)

    def profile(frame, event, arg):
        nonlocal count
        if event == 'call':
            count += 1

    sys.setprofile(profile)
    fib(n)
    sys.setprofile(None)
    return count
