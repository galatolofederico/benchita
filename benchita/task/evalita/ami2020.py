import os
import pandas as pd
from enum import Enum
from benchita.task import ClassficationTaskFromCSV, register_task

class AMI2020Labels(Enum):
    NON_MISOGINO = 0
    MISOGINO = 1

@register_task("ami2020")
class AMI2020(ClassficationTaskFromCSV):
    def __init__(self, config):
        super().__init__(
            csv_file="evalita/ami2020/ami2020.tsv",
            text_column="text",
            label_column="misogynous",
            read_csv_kwargs={"sep": "\t"},
            labels=AMI2020Labels,
            config=config
        )

    @property
    def system(self):
        return f"Il tuo scopo è classificare ciascun tweet come \"Misogino\" o \"Non Misogino\". Riceverai un tweet per volta. Dopo averlo letto, rispondi esclusivamente con l'etichetta appropriata, senza fornire ulteriori spiegazioni o commenti. Le uniche due risposte accettabili sono: \"{AMI2020Labels.NON_MISOGINO.name}\" per i tweet non misogini, e \"{AMI2020Labels.MISOGINO.name}\" per quelli misogini. Assicurati di rispondere solo con una di queste due etichette."
    
    @property
    def inject_confirmation(self):
        return " È tutto chiaro?"
    
    @property
    def inject_confirmation_reply(self):
        return f"Si, sono pronto a procedere con la classificazione. Classificherò ogni tweet come '{AMI2020Labels.NON_MISOGINO.name}' o '{AMI2020Labels.MISOGINO.name}' senza aggiungere commenti. Procediamo."
