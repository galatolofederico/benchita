from benchita.task import get_task

def test_sentipolc():
    ds = get_task("sentipolc")()
    expected_ds0 = {
        "input" : "#Grillo Mi fa paura la gente che urla. Ne abbiamo già visti almeno un paio, ed è finita com'è finita. Niente urla per me, grazie.",
        "output" : "NEGATIVO"
    }
    
    assert ds[0]["input"] == expected_ds0["input"]
    assert ds[0]["output"] == expected_ds0["output"]