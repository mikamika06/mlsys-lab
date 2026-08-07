GRAPHS = [
    {
        "nodes": [
            {"id": "n1", "output_dims": ["batch", "seq", "hidden"]}
        ],
        "constraints": [
            {"dim_name": "seq", "value": 512}
        ]
    },
    {
        "nodes": [
            {"id": "n1", "output_dims": [1, "seq", 768]}
        ],
        "constraints": [
            {"dim_name": "seq", "value": 1024}
        ]
    },
    {
        "nodes": [
            {"id": "n1", "output_dims": ["b", "s", "h"]}
        ],
        "constraints": [
            {"dim_name": "b", "value": 1},
            {"dim_name": "s", "value": 256},
            {"dim_name": "h", "value": 128}
        ]
    }
]

SHAPES_LIST = [
    [1, 32],
    [1, 64],
    [1, 128],
    [1, 256]
]
