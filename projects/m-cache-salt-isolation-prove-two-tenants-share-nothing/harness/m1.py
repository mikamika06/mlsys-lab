import ref


def check(workdir):
    from prefixhash.salt import verify_salt_isolation
    cases = ref.get_test_cases()
    passed = 0
    for blocks_a, blocks_b, salt_a, salt_b in cases:
        got = verify_salt_isolation(blocks_a, blocks_b, salt_a, salt_b)
        want = len({hash((b, salt_a)) for b in blocks_a}.intersection({hash((b, salt_b)) for b in blocks_b})) == 0
        if got == want:
            passed += 1
    return {"salt_isolated": 1.0 if passed == len(cases) else 0.0}
