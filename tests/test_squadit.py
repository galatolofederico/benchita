from benchita.task import get_task

def test_squadIT():
    ds = get_task("squad_it")()
    assert len(ds) == 7609

    expected_ds0 = {
        "input": "La crisi petrolifera del 1973 iniziò nell' ottobre 1973 quando i membri dell' Organizzazione dei Paesi esportatori di petrolio arabo (OAPEC, composta dai membri arabi dell\' OPEC più Egitto e Siria) proclamarono un embargo petrolifero. Alla fine dell' embargo, nel marzo 1974, il prezzo del petrolio era salito da 3 dollari al barile a quasi 12 dollari a livello mondiale; i prezzi americani erano notevolmente più elevati. L' embargo ha causato una crisi petrolifera, o \"shock\", con molti effetti a breve e lungo termine sulla politica globale e sull\' economia globale. Più tardi fu chiamato il \"primo shock petrolifero\", seguito dalla crisi petrolifera del 1979, definita il \"secondo shock petrolifero\".\nQuando è iniziata la crisi petrolifera del 1973?",
        "output": [{'answers': {'text': ['ottobre 1973', 'ottobre 1973', 'ottobre 1973', 'ottobre', '1973'], 'answer_start': [43, 43, 43, 43, 25]}, "id": "5725b33f6a3fe71400b8952d"}]
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
                {'role': 'user', 'content': 'Data la seguente premessa, rispondi alla successiva domanda. La risposta deve essere breve e coincisa. È tutto chiaro?'},
                {'role': 'assistant', 'content': 'Si, sono pronto a rispondere alla domanda. Risponderò in modo breve e coinciso.'},
                {'role': 'user', 'content': "Durante il medioevo, Newcastle era la fortezza inglese più a nord. Annessa inizialmente da Enrico II d'Inghilterra, ricevette un nuovo statuto concesso da Elisabetta I nel 1589, agli inizi dell'età moderna. Una cinta muraria alta circa sette metri e mezzo venne costruita attorno alla città durante il XIII secolo, per difenderla dalle invasioni durante la guerra di frontiera contro la Scozia. Il re scozzese Guglielmo I venne imprigionato in città nel 1174 ed Edoardo I trasportò a sud la Pietra di Scone e William Wallace attraversando Newcastle. La città venne difesa con successo contro gli scozzesi per ben tre volte durante il XIV secolo e venne innalzata al rango di County corporate con un proprio sceriffo da Enrico IV nel 1400.\nQuante volte Newcastle ha combattuto contro gli scozzesi nel XIV secolo?"},
                {'role': 'assistant', 'content': [{'answers': {'text': ['tre volte', 'tre', 'tre volte'], 'answer_start': [613, 613, 613]}, 'id': '572666d9dd62a815002e83b8'}]},
                {'role': 'user', 'content': "Gli studenti dell' Università di Chicago gestiscono oltre 400 club e organizzazioni note come Organizzazioni Studenti Riconosciute (RSO). Questi includono gruppi culturali e religiosi, club e squadre accademiche e organizzazioni di interesse comune. Tra i gruppi extracurriculari si segnalano il Team Bowl dell' University of Chicago College, che ha vinto 118 tornei e 15 campionati nazionali, leader in entrambe le categorie a livello internazionale. Il team competitivo dell' università del Modello delle Nazioni Unite è stato il top in Nord America nel 2013-2014 e 2014-2015. Tra le RSO degne di nota sono la società del cinema studentesco Doc Films, il comitato organizzatore per l' Università di Chicago Scavenger Hunt, il quotidiano studentesco bi-settimanale The Chicago Maroon, il settimanale alternativo South Side Weekly, la seconda più antica compagnia teatrale improvvisata fuori dal campus della nazione, e la radio di proprietà universitaria WHPK.\nQual è il nome della società di film studenteschi più lunga del paese che gestisce continuamente il cinema studentesco?"},
                {'role': 'assistant', 'content': [{'answers': {'text': ['Doc Films', 'Doc Films', 'Doc Films'], 'answer_start': [643, 643, 643]}, 'id': '5728659f4b864d190016498d'}]},
                {'role': 'user', 'content': 'Oltre a Who Wires to Be a Millionaire, il network è entrato negli anni 2000 con i colpi di scena del decennio precedente come The Practice, NYPD Blue e The Wonderful World of Disney e nuove serie come My Wife and Kids e secondo Jim, tutte riuscite ad aiutare ABC rimanere davanti alla concorrenza nella classifica nonostante la partenza successiva di Millionaire. Il 2000 ha visto la fine di "TGIF", che stava lottando per trovare nuovi successi (con Boy Meets World e Sabrina, la strega adolescente, quest\' ultima trasferitasi a The WB nel settembre 2000, cominciando a calare anche da questo punto) dopo la perdita di Family Matters e Step by Step a CBS come parte del suo tentativo fallito di un blocco di commedia orientato alla famiglia venerdì nella stagione 1997-98. Al di fuori di Venerdì stalwart 20/20, venerdì sera è rimasto un punto debole per ABC per i prossimi 11 anni.\nA quale rete Sabrina la Strega adolescente si è trasferita nel 2000?'},
                {'role': 'assistant', 'content': [{'answers': {'text': ['The WB', 'The WB', 'WB'], 'answer_start': [530, 530, 534]}, 'id': '57273b69dd62a815002e99d7'}]},
                {'role': 'user', 'content': 'La crisi petrolifera del 1973 iniziò nell\' ottobre 1973 quando i membri dell\' Organizzazione dei Paesi esportatori di petrolio arabo (OAPEC, composta dai membri arabi dell\' OPEC più Egitto e Siria) proclamarono un embargo petrolifero. Alla fine dell\' embargo, nel marzo 1974, il prezzo del petrolio era salito da 3 dollari al barile a quasi 12 dollari a livello mondiale; i prezzi americani erano notevolmente più elevati. L\' embargo ha causato una crisi petrolifera, o "shock", con molti effetti a breve e lungo termine sulla politica globale e sull\' economia globale. Più tardi fu chiamato il "primo shock petrolifero", seguito dalla crisi petrolifera del 1979, definita il "secondo shock petrolifero".\nQuando è iniziata la crisi petrolifera del 1973?'}
             ],
        'expected': [{'answers': {'text': ['ottobre 1973', 'ottobre 1973', 'ottobre 1973', 'ottobre', '1973'],'answer_start': [43, 43, 43, 43, 25]}, 'id': '5725b33f6a3fe71400b8952d'}]
        }

    ]
    inference_outputs = [[{'answers': {'text': ['ottobre 1973', 'ottobre 1973', 'ottobre 1973', 'ottobre', '1973'],'answer_start': [43, 43, 43, 43, 25]}, 'id': '5725b33f6a3fe71400b8952d'}]]

    inference = []
    for elem, output in zip(inference_inputs, inference_outputs):
        inference.append({
            "messages": elem["messages"],
            "expected": elem["expected"],
            "prompt": elem["messages"][-1]["content"],
            "output": output
        })

    results = ds.evaluate(inference)
    assert results['exact'] == 100.0
    # print(ds.results_summary(results))
