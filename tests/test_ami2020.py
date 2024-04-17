from benchita.task import get_task

def test_ami2020():
    ds = get_task("ami2020")()
    expected_ds0 = {
        "input" : "Aveva voglia di gridare tutta la mia rabbia ma quel maledetto nodo in gola bloccava tutto. Ogni urlo si trasformava in lacrime. Quanto odiava piangere, non lo faceva mai. Guardando le luci della notte si addormentò sfinita con una musica dolce che le faceva da ninna nannaLaila",
        "output" : "NON_MISOGINO"
    }
    
    assert ds[0]["input"] == expected_ds0["input"]
    assert ds[0]["output"] == expected_ds0["output"]

    elem = next(iter(ds.build(num_shots=3, system_style="inject")))
    
    assert "messages" in elem
    assert elem["messages"][-1]["role"] == "user"
    assert elem["messages"][-1]["content"] == expected_ds0["input"]

    inference = []
    inference_outputs = ["NON_MISOGINOdfsdf", "MISOGINO.asddd", "NON_MISOGINO"]
    for _, elem, out in zip(range(0, 3), ds.build(num_shots=3, system_style="inject"), inference_outputs):
        inference.append({
            "messages": elem["messages"],
            "expected": elem["expected"],
            "prompt": elem["messages"][-1]["content"],
            "output": out
        })


    results = ds.evaluate(inference)
    assert results["accuracy"] > 0.5 and results["accuracy"] < 1.0