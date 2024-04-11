from benchita.template import get_template

def test_ironita():
    template = get_template("default")()
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"}
    ]
    assert template.apply_chat_template(messages) == "USER: Hello\nASSISTANT: Hi there\nUSER:"