from keras.models import load_model
from skimage import io
from skimage.transform import resize
from classifier import create_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import random
import matplotlib.pyplot as plt


model = create_model()
model.load_weights('second_try_2.h5')

pts = [os.path.join('data/train/potato', x) for x in os.listdir('data/train/potato')]
npts = [os.path.join('data/train/trash', x) for x in os.listdir('data/train/trash')]
imgs = pts + npts
random.shuffle(imgs)

for img in imgs:
# img = load_img('data/validation/potato/1473498294.jpg')
    img = load_img(img)
    img = img.resize((150, 150))
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    res = model.predict(x)
    print(res)
    plt.imshow(img)
    plt.show()
