import os
import pandas as pd
from enum import Enum
from benchita.task import Task

class IronItaLabel(Enum):
    NON_IRONICO = 0
    IRONICO = 1

class IronIta(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ironita_file = os.path.join(self.base_folder, "ironita", "ironita.csv")
        self.df = pd.read_csv(self.ironita_file, sep=";")

    def _label_to_class(self, label):
        if label == IronItaLabel.IRONICO.value:
            return IronItaLabel.IRONICO.name
        elif label == IronItaLabel.NON_IRONICO.value:
            return IronItaLabel.NON_IRONICO.name
        else:
            raise ValueError("Invalid label")

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        elem = self.df.iloc[idx]
        text = elem["text"]
        label = self._label_to_class(elem["irony"])

        return {
            "input": text,
            "output": label
        }

    @property
    def system(self):
        return f"Il tuo scopo è classificare ciascun tweet come \"Ironico\" o \"Non Ironico\". Riceverai un tweet per volta. Dopo averlo letto, rispondi esclusivamente con l'etichetta appropriata, senza fornire ulteriori spiegazioni o commenti. Le uniche due risposte accettabili sono: \"{IronItaLabel.IRONICO.name}\" per i tweet ironici, e \"{IronItaLabel.NON_IRONICO.name}\" per quelli che non lo sono. Assicurati di rispondere solo con una di queste due etichette."
    
    @property
    def inject_confirmation(self):
        return " È tutto chiaro?"
    
    @property
    def inject_confirmation_reply(self):
        return "Si, sono pronto a procedere con la classificazione. Classificherò ogni tweet come 'IRONICO' o 'NON_IRONICO' senza aggiungere commenti. Procediamo."
