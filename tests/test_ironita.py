from benchita.task import IronIta

def test_ironita():
    ds = IronIta()
    expected_ds0 = {
        "input" : "-Prendere i libri in copisteria-Fare la spesa-Spararmi in bocca-Farmi la doccia",
        "output" : "IRONICO"
    }
    
    assert ds[0]["input"] == expected_ds0["input"]
    assert ds[0]["output"] == expected_ds0["output"]
