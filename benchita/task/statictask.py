import json
import os
import pandas as pd

from benchita.task import Task
from benchita.task import register_task

@register_task("static")
class StaticTask(Task):
    def __init__(self, config):
        super(StaticTask, self).__init__(config)
        if config.num_shots != 0:
            raise ValueError("StaticTask supports only num_shots=0")
        self.fname = os.path.join(self.base_folder, config.args["file"])
        self.ds = json.load(open(self.fname))
        self.max_new_tokens = config.args["max_new_tokens"]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        elem = self.ds[idx]
        return {
            "input": elem["input"],
            "output": elem["output"]
        }

    def evaluate(self, inference):
        correct = 0
        for elem in inference:
            if elem["output"].startswith(elem["expected"]):
                correct += 1

        return dict(
            accuracy=correct / len(inference)
        )   

    def results_summary(self, results):
        accuracy = results["accuracy"]

        return pd.DataFrame({
            "Task": [self.task_name],
            "Accuracy": [accuracy],
        })

    @property
    def system(self):
        return "Rispondi solamente alla seguente domanda senza proseguire la conversazione."

    @property
    def inject_confirmation(self):
        return " È tutto chiaro?"

    @property
    def inject_confirmation_reply(self):
        return "Si, sono pronto a rispondere alla domanda."
    
    @property
    def max_new_tokens(self):
        return self.max_new_tokens

    @max_new_tokens.setter
    def max_new_tokens(self, value):
        self._max_new_tokens = value
