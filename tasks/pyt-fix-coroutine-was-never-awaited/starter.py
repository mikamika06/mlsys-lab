import asyncio

async def delayed_square(n):
    """Square a number asynchronously."""
    await asyncio.sleep(0)
    return n * n

async def delayed_increment(n):
    """Increment a number asynchronously."""
    await asyncio.sleep(0)
    return n + 1

async def collect_results(numbers):
    """Process numbers through the async pipeline and return final values.

    For each number, square it asynchronously, then increment the result
    asynchronously.  Return the list of final values.

    BUG: the returned list contains coroutine objects instead of integers.
    """
    results = []
    for n in numbers:
        squared = delayed_square(n)
        final = delayed_increment(squared)
        results.append(final)
    return results
