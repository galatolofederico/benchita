import os

from benchita.task import Task

class StaticTask(Task):
    def __init__(self, base_folder="./assets"):
        self.base_folder = base_folder

    def __len__(self):
        raise NotImplementedError("__len__ is not implemented")

    def __getitem__(self, idx):
        raise NotImplementedError("__getitem__ is not implemented")

    def evaluate(self, inference_inputs, inference_outputs):
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
