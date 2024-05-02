import json
import pandas as pd

from benchita.task import Task
from benchita.task import register_task

@register_task("static")
class StaticTask(Task):
    def __init__(self, config):
        super(StaticTask, self).__init__(config)
        print(config)
        self.fname = config.args["file"]
        self.ds = json.load(open(self.fname))

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
        f1_macro = results["macro avg"]["f1-score"]

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
        return 64
