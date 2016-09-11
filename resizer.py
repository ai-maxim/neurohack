import os
from skimage.transform import resize
from skimage.io import imread, imsave

LOAD_FOLDER = 'data/raw'
SAVE_FOLDER = 'data/small'
for x in os.listdir(LOAD_FOLDER):
    print(x)
    img = imread(os.path.join(LOAD_FOLDER, x))
    if img.shape[0] < img.shape[1]:
        img = resize(img, (525, 700))
    else:
        img = resize(img, (700, 525))
    imsave(os.path.join(SAVE_FOLDER, x), img)

