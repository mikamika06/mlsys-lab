import asyncio
import types

# ---------------------------------------------------------------------------
# Oracle: the same async helpers, reimplemented independently from the
# learner's file, used to compute the expected answer at grading time.
# ---------------------------------------------------------------------------
async def _oracle_square(n):
    await asyncio.sleep(0)
    return n * n

async def _oracle_increment(n):
    await asyncio.sleep(0)
    return n + 1

async def _oracle_collect(numbers):
    results = []
    for n in numbers:
        squared = await _oracle_square(n)
        final = await _oracle_increment(squared)
        results.append(final)
    return results

# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------
def grade(sol, fx) -> dict:
    """Grade the learner's collect_results against the async oracle.

    Accepts both an async implementation (driven via asyncio.run) and a
    synchronous one that returns the right list directly.
    """
    cases = [
        [1, 2, 3, 4, 5],
        [0],
        [10, 20],
        list(range(10)),
        [42],
    ]

    ok = 1.0
    for nums in cases:
        try:
            expected = asyncio.run(_oracle_collect(list(nums)))

            raw = sol.collect_results(list(nums))

            # The learner may keep it async (normal) or convert to sync.
            if isinstance(raw, types.CoroutineType) or (
                hasattr(raw, "__await__")
            ):
                got = asyncio.run(raw)
            else:
                got = raw
        except Exception:
            ok = 0.0
            break

        if got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
