import datasets
import enum
from benchita.task import SquadV2Task, register_task


@register_task("squad")
class SquadIT(SquadV2Task):
    def __init__(self):
        super().__init__()
        self.ds = datasets.load_dataset("squad_it")["test"]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        elem = self.ds[idx]

        context = elem['context']
        question = elem['question']
        text = context + "\n" + question

        # Warning - tokenizer.apply_chat_template does not support expected as a list!
        answers = elem['answers']['text'][0]

        return {
            "id": elem["id"],
            "answer_start": elem["answers"]["answer_start"][0],
            "input": text,
            "output": answers
        }
        # id = elem["id"]
        # context = elem['context']
        # question = elem['question']
        # answers = [{"answers": elem['answers']}]
        # answers = [{**answer, "id": elem["id"]} for answer in answers]
        #
        # text = context + "\n" + question
        # return {
        #     "id": id,
        #     "input": text,
        #     "output": answers
        # }

    @property
    def system(self):
        return "Data la seguente premessa, rispondi alla successiva domanda. La risposta deve essere breve e coincisa."

    @property
    def inject_confirmation(self):
        return " È tutto chiaro?"

    @property
    def inject_confirmation_reply(self):
        return "Si, sono pronto a rispondere alla domanda. Risponderò in modo breve e coinciso."

    @property
    def max_new_tokens(self):
        return 100

