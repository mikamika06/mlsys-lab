def branch_mispredict_counts(arr_sorted, arr_unsorted, threshold):
    def count(values):
        state = 1
        misses = 0

        for x in values:
            taken = x > threshold
            prediction = state >= 2

            if prediction != taken:
                misses += 1

            if taken:
                state = min(3, state + 1)
            else:
                state = max(0, state - 1)

        return misses

    return count(arr_sorted), count(arr_unsorted)
