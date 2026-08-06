from .lexer import TemplateError, tokenize


def parse(src):
    """The token stream as a node tree.

    Nodes you need: literal text, an action holding a pipeline, if/else if/else,
    range with zero, one or two loop variables, with, and variable assignment
    (`$x := v` defines, `$x = v` assigns to an existing one).

    A pipeline is commands separated by `|`; each command is a function or a
    value followed by its arguments, and the previous stage arrives as the last
    argument. Operands: `.`, `.Field.Sub`, `$`, `$var`, `$var.Field`, quoted and
    raw strings, numbers, true/false/nil, a parenthesised pipeline, or a
    function name.
    """
    raise NotImplementedError
