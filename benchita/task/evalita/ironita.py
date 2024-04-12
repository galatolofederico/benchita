import os
import pandas as pd
from enum import Enum
from benchita.task import ClassficationTaskFromCSV, register_task

class IronItaLabel(Enum):
    NON_IRONICO = 0
    IRONICO = 1

@register_task("ironita")
class IronIta(ClassficationTaskFromCSV):
    def __init__(self, *args, **kwargs):
        super().__init__(
            csv_file="evalita/ironita/ironita.csv",
            text_column="text",
            label_column="irony",
            read_csv_kwargs={"sep": ";"},
            labels=IronItaLabel,
        )

    @property
    def system(self):
        return f"Il tuo scopo è classificare ciascun tweet come \"Ironico\" o \"Non Ironico\". Riceverai un tweet per volta. Dopo averlo letto, rispondi esclusivamente con l'etichetta appropriata, senza fornire ulteriori spiegazioni o commenti. Le uniche due risposte accettabili sono: \"{IronItaLabel.IRONICO.name}\" per i tweet ironici, e \"{IronItaLabel.NON_IRONICO.name}\" per quelli che non lo sono. Assicurati di rispondere solo con una di queste due etichette."
    
    @property
    def inject_confirmation(self):
        return " È tutto chiaro?"
    
    @property
    def inject_confirmation_reply(self):
        return f"Si, sono pronto a procedere con la classificazione. Classificherò ogni tweet come '{IronItaLabel.IRONICO.name}' o '{IronItaLabel.NON_IRONICO.name}' senza aggiungere commenti. Procediamo."
