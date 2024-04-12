import os
import pandas as pd
from enum import Enum
from benchita.task import ClassficationTaskFromCSV, register_task

class SentipolcLabel(Enum):
    POSITIVO = 0
    NEGATIVO = 1

@register_task("sentipolc")
class Sentipolc(ClassficationTaskFromCSV):
    def __init__(self, *args, **kwargs):
        super().__init__(
            csv_file="evalita/sentipolc/sentipolc.csv",
            text_column="tweet",
            label_column="negativo",
            read_csv_kwargs={"sep": ";"},
            labels=SentipolcLabel,
        )

    @property
    def system(self):
        return f"Il tuo scopo è classificare ciascun tweet come \"Positivo\" o \"Negativo\". Riceverai un tweet per volta. Dopo averlo letto, rispondi esclusivamente con l'etichetta appropriata, senza fornire ulteriori spiegazioni o commenti. Le uniche due risposte accettabili sono: \"{SentipolcLabel.POSITIVO.name}\" per i tweet positivi, e \"{SentipolcLabel.NEGATIVO.name}\" per quelli negativi. Assicurati di rispondere solo con una di queste due etichette."
    
    @property
    def inject_confirmation(self):
        return " È tutto chiaro?"
    
    @property
    def inject_confirmation_reply(self):
        return f"Si, sono pronto a procedere con la classificazione. Classificherò ogni tweet come '{SentipolcLabel.POSITIVO.name}' o '{SentipolcLabel.NEGATIVO.name}' senza aggiungere commenti. Procediamo."
