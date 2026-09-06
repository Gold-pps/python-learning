from hello import hello

def test_default():
    assert hello()=="hello Gold_pps"

def test_argument():
    for name in ["A","B","C"]:
        assert hello(name)== f"hello {name}"