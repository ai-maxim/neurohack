import os
import random

TRAIN_PATH = 'data/train/potato'
VALIDATION_PATH = 'data/validation/potato'

imgs = os.listdir(TRAIN_PATH)

r = set()
for x in range(len(imgs) // 10):
    r.add(random.choice(imgs))

print(r)
for x in r:
    os.rename(
        os.path.join(TRAIN_PATH, x),
        os.path.join(VALIDATION_PATH, x)
    )



