SUPPORTED = ["abs", "length", "map", "join", "select", "reject"]

CONFIGS = [
    {
        "template": "{{ messages | map(attribute='content') | join('\n') }}",
        "unsupported": []
    },
    {
        "template": "{{ messages | unknown_filter | length }}",
        "unsupported": ["unknown_filter"]
    },
    {
        "template": "{{ a | foo | bar | abs }}",
        "unsupported": ["bar", "foo"]
    }
]
