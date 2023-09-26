import os

from src.anpr_module.predict import run

model = '../lib/best.pt'
source = '../lib/EWP05W.jpg'

reg = run(src=source, model=model)
assert reg == 'EWP05W'
