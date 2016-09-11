import cv2
from os.path import join
from keras.models import load_model
from skimage import io
from skimage.transform import resize
from classifier import create_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import random
import numpy as np
import matplotlib.pyplot as plt


PREVIEW = "preview"

OUTPUT_FOLDER = 'data/yellow_potato'


print("loading model....")
model = create_model()
model.load_weights('second_try_2.h5')


def main():
    print("preparing opencv...")
    cv2.namedWindow(PREVIEW)
    vc = cv2.VideoCapture(0)
    i = 0

    if vc.isOpened():
        has_frame, frame = vc.read()
    else:
        has_frame = False

    print("loop")
    while has_frame:
        has_frame, frame = vc.read()
        # <type 'tuple'>: (480, 640, 3)
        frame = do(frame)

        cv2.imshow(PREVIEW, frame)

        key = cv2.waitKey(20)
        if key == 115: # s
            filename = join(OUTPUT_FOLDER, "{}.png".format(i))
            cv2.imwrite(filename, frame)
            print("saved: " + filename)
            i+=1

        if key == 27: # esc
            break
    cv2.destroyWindow(PREVIEW)


def do(frame):
    # rf = cv2.cvtColor(rf, cv2.COLOR_BGR2GRAY)
    for_net = cv2.resize(frame, (150, 150))
    for_net = np.swapaxes(for_net, 0, 2)
    for_net = for_net.reshape((1,) + for_net.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    res = model.predict(for_net)
    print(res)

    pass
    # res = net1.predict(rf.reshape(1,9216) / 255)[0]
    # res = net1.predict(rf.reshape(1,1,96,96) / 255)[0]
    # for x,y in zip(res[0::2]*48+48, res[1::2]*48+48):
    #     cv2.drawMarker(rf, (int(x), int(y)), 255)

    return frame


if __name__ == "__main__":
    main()
