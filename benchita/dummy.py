from benchita.task import Task, ClassificationTask
import random
import string
from tqdm import tqdm

class DummyModel():
    def __init__(self, task):
        self.task = task
        if isinstance(task, Task):
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
            elif isinstance(elem["expected"], list):
                output = random.choice(elem["expected"])
            else:
                raise Exception("Invalid expected type")
            
            if random.random() < 0.3:
                output += "".join(random.choices(string.ascii_lowercase, k=10))

            return output

    def simulate_inference(self, dataset):
        ret = []
        for elem in tqdm(dataset):
            ret.append({
                "messages": elem["messages"],
                "expected": elem["expected"],
                "input": elem["prompt"],
                "output": self.simulate_output(elem)
            })
        return ret