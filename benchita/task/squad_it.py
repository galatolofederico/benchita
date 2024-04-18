import datasets
import enum
from benchita.task import ComprehensionTask, register_task


@register_task("squad_it")
class SquadIT(ComprehensionTask):
    def __init__(self):
        super().__init__()
        self.ds = datasets.load_dataset("squad_it")["test"]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        elem = self.ds[idx]
        id = elem["id"]
        context = elem['context']
        question = elem['question']
        answers = [{"answers": elem['answers']}]
        answers = [{**answer, "id": elem["id"]} for answer in answers]

        text = context + "\n" + question
        return {
            "id": id,
            "input": text,
            "output": answers
        }

    @property
    def system(self):
        return "Data la seguente premessa, rispondi alla successiva domanda. La risposta deve essere breve e coincisa."

    @property
    def inject_confirmation(self):
        return " È tutto chiaro?"

    @property
    def inject_confirmation_reply(self):
        return "Si, sono pronto a rispondere alla domanda. Risponderò in modo breve e coinciso."
