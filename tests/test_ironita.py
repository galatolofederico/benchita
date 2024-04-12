from benchita.task import get_task

def test_ironita():
    ds = get_task("ironita")()
    expected_ds0 = {
        "input" : "-Prendere i libri in copisteria-Fare la spesa-Spararmi in bocca-Farmi la doccia",
        "output" : "IRONICO"
    }
    
    assert ds[0]["input"] == expected_ds0["input"]
    assert ds[0]["output"] == expected_ds0["output"]

    elem = next(iter(ds.build(num_shots=3, system_style="inject")))
    
    assert "messages" in elem
    assert elem["messages"][-1]["role"] == "user"
    assert elem["messages"][-1]["content"] == expected_ds0["input"]

    inference_inputs = [{'messages': [{'role': 'user', 'content': 'Il tuo scopo è classificare ciascun tweet come "Ironico" o "Non Ironico". Riceverai un tweet per volta. Dopo averlo letto, rispondi esclusivamente con l\'etichetta appropriata, senza fornire ulteriori spiegazioni o commenti. Le uniche due risposte accettabili sono: "IRONICO" per i tweet ironici, e "NON_IRONICO" per quelli che non lo sono. Assicurati di rispondere solo con una di queste due etichette. È tutto chiaro?'}, {'role': 'assistant', 'content': "Si, sono pronto a procedere con la classificazione. Classificherò ogni tweet come 'IRONICO' o 'NON_IRONICO' senza aggiungere commenti. Procediamo."}, {'role': 'user', 'content': '<MENTION_1> Mira quin vi he trobat a Roma... <URL>'}, {'role': 'assistant', 'content': 'NON_IRONICO'}, {'role': 'user', 'content': 'Ma se gli stranieri ci pagano le pensioni, gli italiani cosa cazzo lavorano da fare? #DallaVostraParte'}, {'role': 'assistant', 'content': 'IRONICO'}, {'role': 'user', 'content': '<MENTION_1> la buona scuola è quello che è accaduto al test delle #specializzazionimediche ?Almeno prendete una posizione netta in merito'}, {'role': 'assistant', 'content': 'IRONICO'}, {'role': 'user', 'content': '-Prendere i libri in copisteria-Fare la spesa-Spararmi in bocca-Farmi la doccia'}], 'expected': 'IRONICO', 'prompt': '<bos><start_of_turn>user\nIl tuo scopo è classificare ciascun tweet come "Ironico" o "Non Ironico". Riceverai un tweet per volta. Dopo averlo letto, rispondi esclusivamente con l\'etichetta appropriata, senza fornire ulteriori spiegazioni o commenti. Le uniche due risposte accettabili sono: "IRONICO" per i tweet ironici, e "NON_IRONICO" per quelli che non lo sono. Assicurati di rispondere solo con una di queste due etichette. È tutto chiaro?<end_of_turn>\n<start_of_turn>model\nSi, sono pronto a procedere con la classificazione. Classificherò ogni tweet come \'IRONICO\' o \'NON_IRONICO\' senza aggiungere commenti. Procediamo.<end_of_turn>\n<start_of_turn>user\n<MENTION_1> Mira quin vi he trobat a Roma... <URL><end_of_turn>\n<start_of_turn>model\nNON_IRONICO<end_of_turn>\n<start_of_turn>user\nMa se gli stranieri ci pagano le pensioni, gli italiani cosa cazzo lavorano da fare? #DallaVostraParte<end_of_turn>\n<start_of_turn>model\nIRONICO<end_of_turn>\n<start_of_turn>user\n<MENTION_1> la buona scuola è quello che è accaduto al test delle #specializzazionimediche ?Almeno prendete una posizione netta in merito<end_of_turn>\n<start_of_turn>model\nIRONICO<end_of_turn>\n<start_of_turn>user\n-Prendere i libri in copisteria-Fare la spesa-Spararmi in bocca-Farmi la doccia<end_of_turn>\n<start_of_turn>model\n'}, {'messages': [{'role': 'user', 'content': 'Il tuo scopo è classificare ciascun tweet come "Ironico" o "Non Ironico". Riceverai un tweet per volta. Dopo averlo letto, rispondi esclusivamente con l\'etichetta appropriata, senza fornire ulteriori spiegazioni o commenti. Le uniche due risposte accettabili sono: "IRONICO" per i tweet ironici, e "NON_IRONICO" per quelli che non lo sono. Assicurati di rispondere solo con una di queste due etichette. È tutto chiaro?'}, {'role': 'assistant', 'content': "Si, sono pronto a procedere con la classificazione. Classificherò ogni tweet come 'IRONICO' o 'NON_IRONICO' senza aggiungere commenti. Procediamo."}, {'role': 'user', 'content': '\\"io ho paura dei musulmani, degli immigrati, degli zingari e non vorrei che ci fossero in Italia ma non sono razzista\\"certamente'}, {'role': 'assistant', 'content': 'IRONICO'}, {'role': 'user', 'content': 'Che bella Europa non solo ci lascia soli davanti alla tragedia dei migranti costringendoci ad indebitarci per salvarli ed assisterli grazie'}, {'role': 'assistant', 'content': 'IRONICO'}, {'role': 'user', 'content': 'Ma quelli che dato che sei italiana anche se in un paese straniero ti consigliano una pizzeria buonissima... <MENTION_1> <MENTION_2>'}, {'role': 'assistant', 'content': 'IRONICO'}, {'role': 'user', 'content': '...comunque con una crociera Costa se non ti ammazza Schettino prima ti spara il terrorista dopo...'}], 'expected': 'IRONICO', 'prompt': '<bos><start_of_turn>user\nIl tuo scopo è classificare ciascun tweet come "Ironico" o "Non Ironico". Riceverai un tweet per volta. Dopo averlo letto, rispondi esclusivamente con l\'etichetta appropriata, senza fornire ulteriori spiegazioni o commenti. Le uniche due risposte accettabili sono: "IRONICO" per i tweet ironici, e "NON_IRONICO" per quelli che non lo sono. Assicurati di rispondere solo con una di queste due etichette. È tutto chiaro?<end_of_turn>\n<start_of_turn>model\nSi, sono pronto a procedere con la classificazione. Classificherò ogni tweet come \'IRONICO\' o \'NON_IRONICO\' senza aggiungere commenti. Procediamo.<end_of_turn>\n<start_of_turn>user\n\\"io ho paura dei musulmani, degli immigrati, degli zingari e non vorrei che ci fossero in Italia ma non sono razzista\\"certamente<end_of_turn>\n<start_of_turn>model\nIRONICO<end_of_turn>\n<start_of_turn>user\nChe bella Europa non solo ci lascia soli davanti alla tragedia dei migranti costringendoci ad indebitarci per salvarli ed assisterli grazie<end_of_turn>\n<start_of_turn>model\nIRONICO<end_of_turn>\n<start_of_turn>user\nMa quelli che dato che sei italiana anche se in un paese straniero ti consigliano una pizzeria buonissima... <MENTION_1> <MENTION_2><end_of_turn>\n<start_of_turn>model\nIRONICO<end_of_turn>\n<start_of_turn>user\n...comunque con una crociera Costa se non ti ammazza Schettino prima ti spara il terrorista dopo...<end_of_turn>\n<start_of_turn>model\n'}]
    inference_outputs = ['IRONICO.trash', 'NON_IRONICO']


    inference = []
    for elem, output in zip(inference_inputs, inference_outputs):
        inference.append({
            "messages": elem["messages"],
            "expected": elem["expected"],
            "prompt": elem["prompt"],
            "output": output
        })

    ds.evaluate(inference)