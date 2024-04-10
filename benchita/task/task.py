import random
from collections.abc import Sequence

class Task(Sequence):
    def __init__(self, base_folder="./assets"):
        self.base_folder = base_folder

    @property
    def system(self):
        raise NotImplementedError

    @property
    def inject_confirmation(self):
        raise NotImplementedError
    
    @property
    def inject_confirmation_reply(self):
        raise NotImplementedError

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
            
            yield messages