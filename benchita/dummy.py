from benchita.task import Task, ClassificationTask, SquadV2Task
import random
import string
from tqdm import tqdm

class DummyModel():
    def __init__(self, task):
        self.task = task
        if isinstance(task, ClassificationTask):
            self.task_type = "classification"
        else:
            self.task_type = "generic"
    
    def simulate_output(self, elem):
        if self.task_type == "classification":
            output = random.choice(list(self.task.classes))
            if random.random() < 0.3:
                output += "".join(random.choices(string.ascii_lowercase, k=10))
            return output
        elif self.task_type == "generic":
            if isinstance(elem["expected"], str):
                output = elem["expected"]
            # Warning - tokenizer.apply_chat_template does not support expected as a list!
            elif isinstance(elem["expected"], list):
                output = random.choice(elem["expected"])
            else:
                raise Exception("Invalid expected type")
            
            if random.random() < 0.3:
                output += "".join(random.choices(string.ascii_lowercase, k=10))

            return output

    def simulate_inference(self, dataset):
        ret = []
        for elem in tqdm(dataset, desc="Inference"):
            d = {
                "messages": elem["messages"],
                "expected": elem["expected"],
                "input": elem["prompt"],
                "model_input": elem["model_input"],
                "output": self.simulate_output(elem)
            }
            if isinstance(self.task, SquadV2Task):
                d["id"] = elem["id"]
                d["answer_start"] = elem["answer_start"]

            ret.append(d)
        return ret