CASES = [
    {
        "draft": [10, 20, 30, 40, 50],
        "vocab": {10: 100, 20: 200, 30: 300, 40: 400, 50: 500},
        "expected": [100, 200, 300, 400, 500],
        "bytes_orig": b"hello",
        "bytes_real": b"hello"
    },
    {
        "draft": [11, 22, 33, 44],
        "vocab": {11: 110, 22: 220, 33: 330, 44: 440},
        "expected": [110, 220, 330, 440],
        "bytes_orig": b"test",
        "bytes_real": b"test"
    },
    {
        "draft": [5, 10, 15, 20],
        "vocab": {5: 50, 10: 100, 15: 150, 20: 200},
        "expected": [50, 100, 150, 200],
        "bytes_orig": b"data",
        "bytes_real": b"data"
    },
    {
        "draft": [1, 2, 3, 4, 5, 6],
        "vocab": {1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60},
        "expected": [10, 20, 30, 40, 50, 60],
        "bytes_orig": b"abcdef",
        "bytes_real": b"abcdef"
    },
    {
        "draft": [7, 8, 9],
        "vocab": {7: 70, 8: 80, 9: 90},
        "expected": [70, 80, 90],
        "bytes_orig": b"xyz",
        "bytes_real": b"xyz"
    }
]
