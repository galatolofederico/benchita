import datasets
import enum
from benchita.task import ClassificationTask, register_task


@register_task("xcopa")
class XCopa(ClassificationTask):
    def __init__(self):
        super().__init__()
        self.ds = datasets.load_dataset("xcopa", "it")["test"]

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        elem = self.ds[idx]
        tipo = "Indicare la causa." if elem["question"] == "cause" else "Indicare l'effetto."
        text = f"Premessa: {elem['premise']} {tipo}\nA){elem['choice1']}\nB){elem['choice2']}"
        
        return {
            "input": text,
            "output": "A" if elem["label"] == 0 else "B"
        }
    
    @property
    def classes(self):
        return ["A", "B"]
    
    @property
    def max_new_tokens(self):
        return 3
    
    @property
    def system(self):
        return "Dati una premessa e due possibili opzioni (A e B), determina quale delle due opzioni è la causa o l'effetto più probabile data la premessa. Rispondi con \"A\" se l'opzione A è la più logica o probabile data la premessa, oppure con \"B\" se è  l'opzione B. Le uniche risposte accettabili sono A e B. Assicurati di rispondere solo con una di queste due etichette. Non fornire ulteriori spiegazioni o commenti."
    
    @property
    def inject_confirmation(self):
        return " È tutto chiaro?"
    
    @property
    def inject_confirmation_reply(self):
        return "Si, sono pronto a procedere con la classificazione. Risolverò il problema rispondendo con 'A' o 'B' senza aggiungere commenti. Procediamo."