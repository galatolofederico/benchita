_template_registry = dict()

class Template:
    def __init__(self, *args, **kwargs):
        pass

    @property
    def apply_chat_template(self, messages, **kwargs):
        raise NotImplementedError


def register_template(name):
    def decorator(cls):
        if name in _template_registry:
            raise ValueError(f"Template {name} already exists")
        _template_registry[name] = cls
        return cls
    return decorator

def get_template(name):
    if name not in _template_registry:
        raise ValueError(f"Template {name} not found")
    return _template_registry[name]

def get_templates():
    return list(_template_registry.keys())