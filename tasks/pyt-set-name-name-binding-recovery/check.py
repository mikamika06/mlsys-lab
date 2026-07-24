def grade(sol, fx) -> dict:
    try:
        NF = sol.NamedField

        class Point:
            x = NF()
            y = NF()

        expected1 = {"x": "x", "y": "y"}
        got1 = sol.recovered_names(Point)

        # The classic __set_name__ gotcha: the SAME descriptor instance
        # bound to two different class attributes only gets to remember
        # one name — __set_name__ fires once per binding, in class-body
        # order, so the LAST binding wins and both attributes end up
        # reporting that same recovered name.
        shared = NF()

        class Weird:
            a = shared
            b = shared

        expected2 = {"a": "b", "b": "b"}
        got2 = sol.recovered_names(Weird)

        # non-descriptor class attributes (constants, methods) must be
        # excluded from the recovered mapping entirely
        class Mixed:
            CONST = 5

            def method(self):
                return self.CONST

            p = NF()

        expected3 = {"p": "p"}
        got3 = sol.recovered_names(Mixed)

        # round-trip through get/set must also work, keyed by the
        # recovered private name
        pt = Point()
        pt.x = 3
        pt.y = 4
        roundtrip_ok = (pt.x, pt.y) == (3, 4) and "x" not in pt.__dict__

        ok = 1.0 if (
            got1 == expected1
            and got2 == expected2
            and got3 == expected3
            and roundtrip_ok
        ) else 0.0
    except Exception:
        ok = 0.0

    return {"exact_match": ok}
