import os

from anprmodule.predict import run

def test__end_to_end():
    model = '../lib/best.pt'
    source = '../lib/EWP05W.jpg'

    reg = run(src=source, model=model)
    assert reg == 'EWP05W'
    assert isinstance(reg, str)


def run_tests():
    test__end_to_end()


if __name__=="__main__":
    run_tests()
