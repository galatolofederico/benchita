from collections.abc import Sequence

class Task(Sequence):
    def __init__(self):
        pass

    def sample(self, num):
        raise NotImplementedError

    def system(self):
        raise NotImplementedError

    def inject_confirmation(self):
        raise NotImplementedError
    
    def inject_confirmation_reply(self):
        raise NotImplementedError

    def get_shots(self, num_shots):
        shots = self.sample(num_shots)
