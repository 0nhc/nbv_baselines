EXTERNAL_FREEZE_MODULES = set()

def external_freeze(cls):
    if not hasattr(cls, 'load') or not callable(getattr(cls, 'load')):
        raise TypeError(f"external module <{cls.__name__}> must implement a 'load' method")
    EXTERNAL_FREEZE_MODULES.add(cls)
    return cls