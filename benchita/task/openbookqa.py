import datasets
from benchita.task import ClassificationTask, register_task

_option_labels = ["A", "B", "C", "D"]

@register_task("openbook")
class OpenbookQA(ClassificationTask):
    def __init__(self):
        super().__init__()
        self.ds = datasets.load_dataset("yuri-no/openbookqa-ITA")["test"]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        elem = self.ds[idx]

        question = elem['question_stem']
        options = [label + ": " + answer for label, answer in zip(_option_labels, elem['choices']['text'])]
        options2 = '\n'.join(options)
        text = question + "\n" + options2

        return {
            "input": text,
            "output": elem['answerKey']
        }


    @property
    def classes(self):
        return ["A", "B", "C", "D"]

    @property
    def system(self):
        return ("Riceverai una frase per volta con quattro possibili risposte (\"A\", \"B\", \"C\", \"D\"). "
                "Il tuo scopo è scegliere la risposta corretta tra le quattro fornite. "
                "Rispondi esclusivamente con l'etichetta appropriata, senza fornire ulteriori spiegazioni o commenti. ")

    @property
    def inject_confirmation(self):
        return " È tutto chiaro?"

    @property
    def inject_confirmation_reply(self):
        return ("Si, sono pronto a procedere con la classificazione. "
                "Risponderò soltanto con l'opzione corretta tra 'A', 'B', 'C' o 'D' senza aggiungere commenti. Procediamo.")

    @property
    def max_new_tokens(self):
        return 2
