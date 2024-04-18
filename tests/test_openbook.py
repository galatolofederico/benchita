from benchita.task import get_task
from benchita.task import openbookqa

def test_openbook():
    ds = get_task("openbook")()
    assert len(ds) == 500

    expected_ds0 = {
        "input": "Una persona desidera iniziare a risparmiare denaro per potersi permettere una bella vacanza alla fine dell'anno. Dopo aver esaminato il proprio budget e le proprie spese, decidono che il modo migliore per risparmiare denaro è\n"
                 "A: fare più telefonate\n"
                 "B: smettere di pranzare fuori\n"
                 "C: acquistare meno con soldi del monopoli\n"
                 "D: andare a pranzo con gli amici",
        "output": "B"
    }

    assert ds[0]["input"] == expected_ds0["input"]
    assert ds[0]["output"] == expected_ds0["output"]

    elem = next(iter(ds.build(num_shots=3, system_style="inject")))
    assert "messages" in elem
    assert elem["messages"][-1]["role"] == "user"
    assert elem["messages"][-1]["content"] == expected_ds0["input"]

    inference_inputs = [
        {'messages':
            [
                {'role': 'user', 'content': 'Riceverai una frase per volta con quattro possibili risposte ("A", "B", "C", "D"). Il tuo scopo è scegliere la risposta corretta tra le quattro fornite. Rispondi esclusivamente con l\'etichetta appropriata, senza fornire ulteriori spiegazioni o commenti.  È tutto chiaro?'},
                {'role': 'assistant', 'content': "Si, sono pronto a procedere con la classificazione. Risponderò soltanto con l'opzione corretta tra 'A', 'B', 'C' o 'D' senza aggiungere commenti. Procediamo."},
                {'role': 'user', 'content': "Se il tuo cane è in sovrappeso\nA: aggiungi più grassi alla loro dieta\nB: riduci l'apporto calorico\nC: fai dormire di più\nD: aumenta il loro apporto calorico"},
                {'role': 'assistant', 'content': 'B'},
                {'role': 'user', 'content': "Un emisfero sperimenta l'estate quando\nA: è inclinato verso Giove\nB: è inclinato verso la luna\nC: è inclinato verso la stella più grande del sistema solare\nD: gira in senso antiorario sull'asse terrestre"},
                {'role': 'assistant', 'content': 'C'},
                {'role': 'user', 'content': 'La fotosintesi fa cosa convertendo anidride carbonica, acqua e luce solare in carboidrati?\nA: nutre piccoli pezzi di proteine che hanno bisogno di mangiare con piccoli frullati\nB: fornendo nutrimento che consente una certa crescita alla vegetazione\nC: mescola i carboidrati nella materia vegetale solubile\nD: produce una buona proteina vegetale'},
                {'role': 'assistant', 'content': 'B'},
                {'role': 'user', 'content': "Una persona desidera iniziare a risparmiare denaro per potersi permettere una bella vacanza alla fine dell'anno. Dopo aver esaminato il proprio budget e le proprie spese, decidono che il modo migliore per risparmiare denaro è\nA: fare più telefonate\nB: smettere di pranzare fuori\nC: acquistare meno con soldi del monopoli\nD: andare a pranzo con gli amici"}
            ],
            'expected': 'B'
        }
    ]
    inference_outputs = ["B", "B", "C", "D"]

    inference = []
    for elem, output in zip(inference_inputs, inference_outputs):
        inference.append({
            "messages": elem["messages"],
            "expected": elem["expected"],
            "prompt": elem["messages"][-1]["content"],
            "output": output
        })
    results = ds.evaluate(inference)
    # print(results)
    assert results['accuracy'] == 1.0



module = openbookqa.OpenbookQA()
test_openbook()