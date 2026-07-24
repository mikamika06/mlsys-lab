import sys


def header_byte_map(objects):
    header = 16
    return [
        [
            int(sys.getsizeof(obj)),
            header,
            int(sys.getsizeof(obj)) - header,
        ]
        for obj in objects
    ]
