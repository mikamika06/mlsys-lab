from itertools import permutations


def reorder_bases(bases):
    def works(order):
        try:
            class Combined(*order):
                def run(self):
                    return super().run()
            result = Combined().run()
        except Exception:
            return False
        return result[:len(order)] == [cls._tag for cls in order] and "Root" in result

    for order in permutations(bases):
        if works(order):
            return tuple(order)
    raise TypeError("No cooperative ordering exists")
