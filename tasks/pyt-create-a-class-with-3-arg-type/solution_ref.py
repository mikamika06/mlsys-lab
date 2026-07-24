def build_animal_hierarchy() -> tuple:
    """Build (Animal, Dog) using only the 3-argument type(name, bases, ns).

    Animal: base (object,), kind='animal', speak() -> "{kind} makes a sound".
    Dog: base (Animal,), kind='dog', speak() -> "{kind} barks".
    """
    def _animal_speak(self):
        return f"{self.kind} makes a sound"

    Animal = type("Animal", (object,), {"kind": "animal", "speak": _animal_speak})

    def _dog_speak(self):
        return f"{self.kind} barks"

    Dog = type("Dog", (Animal,), {"kind": "dog", "speak": _dog_speak})

    return Animal, Dog
