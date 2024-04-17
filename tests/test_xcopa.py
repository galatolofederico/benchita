from benchita.task import get_task

def test_xcopa():
    ds = get_task("xcopa")()
    expected_ds0 = {'input': "Premessa: L'oggetto era incartato nella plastica bollata. Indicare la causa.\nA)Era delicato.\nB)Era piccolo.", 'output': 'A'}
    
    assert ds[0]["input"] == expected_ds0["input"]
    assert ds[0]["output"] == expected_ds0["output"]

    inference = []
    inference_outputs = ["A", "AB", "BB", "B"]
    for _, elem, out in zip(range(0, 43), ds.build(num_shots=3, system_style="inject"), inference_outputs):
        inference.append({
            "messages": elem["messages"],
            "expected": elem["expected"],
            "prompt": elem["messages"][-1]["content"],
            "output": out
        })

    results = ds.evaluate(inference)
    assert results["accuracy"] == 0.75