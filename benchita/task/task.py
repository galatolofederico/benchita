import random
from collections.abc import Sequence

class Task(Sequence):
    def __init__(self, base_folder="./assets"):
        self.base_folder = base_folder

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    @property
    def system(self):
        raise NotImplementedError

    @property
    def inject_confirmation(self):
        raise NotImplementedError
    
    @property
    def inject_confirmation_reply(self):
        raise NotImplementedError
    
    @property
    def max_new_tokens(self):
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
            
            messages.append(dict(
                role="user",
                content=current["input"]
            ))

            yield dict(messages=messages, expected=current["output"])

    def evaluate(self, inference_inputs, inference_outputs):
        raise NotImplementedError