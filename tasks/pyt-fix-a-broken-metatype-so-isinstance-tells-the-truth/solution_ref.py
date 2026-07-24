def make_truthful_type(name, accepted_type):
    class TruthfulMeta(type):
        def __instancecheck__(cls, instance):
            return isinstance(instance, accepted_type)

    return TruthfulMeta(name, (), {})
