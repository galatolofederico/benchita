import random
from collections.abc import Sequence

_task_registry = dict()

class Task(Sequence):
    def __init__(self, config, base_folder="./assets"):
        self._config = config
        self.base_folder = base_folder

    def __len__(self):
        raise NotImplementedError("__len__ is not implemented")

    def __getitem__(self, idx):
        raise NotImplementedError("__getitem__ is not implemented")

    def evaluate(self, inference):
        raise NotImplementedError("evaluate is not implemented")
    
    def results_summary(self, results):
        raise NotImplementedError("results_summary is not implemented")

    @property
    def system(self):
        raise NotImplementedError("system is not implemented")

    @property
    def inject_confirmation(self):
        raise NotImplementedError("inject_confirmation not implemented")
    
    @property
    def inject_confirmation_reply(self):
        raise NotImplementedError("inject_confermation_reply is not implemented")
    
    @property
    def max_new_tokens(self):
        raise NotImplementedError("max_new_tokens is not implemented")

    @property
    def task_name(self):
        for k, v in _task_registry.items():
            if isinstance(self, v):
                return k
        return "UNKNOWN_TASK"

    def build(self, *, num_shots, system_style):
        assert system_style in ["system", "inject"]
        for i in range(0, len(self)):
            current = self[i]
            random_indexes = random.sample(range(0, len(self)), num_shots)
            while i in random_indexes:
                random_indexes = random.sample(range(0, len(self)), num_shots)
            
            messages = []
            if system_style == "system":
                messages.append(dict(
                    role="system",
                    content=self.system
                ))
            else:
                messages.append(dict(
                    role="user",
                    content=self.system + self.inject_confirmation
                ))
                messages.append(dict(
                    role="assistant",
                    content=self.inject_confirmation_reply
                ))
            
            for idx in random_indexes:
                messages.append(dict(
                    role="user",
                    content=self[idx]["input"]
                ))
                messages.append(dict(
                    role="assistant",
                    content=self[idx]["output"]
                ))
            
            messages.append(dict(
                role="user",
                content=current["input"]
            ))
            out = dict(messages=messages, expected=current["output"], model_input=current["input"])

            try:
                out["id"] = current["id"]
                out["answer_start"] = current["answer_start"]
            except KeyError:
                pass
            yield out

    

def register_task(name):
    def decorator(cls):
        if name in _task_registry:
            raise ValueError(f"Task {name} already exists")
        _task_registry[name] = cls
        cls.task_name = name
        return cls
    return decorator

def get_task(name):
    if name not in _task_registry:
        raise ValueError(f"Task {name} not found")
    return _task_registry[name]

def get_tasks():
    return list(_task_registry.keys())