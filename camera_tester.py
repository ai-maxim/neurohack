import cv2
from os.path import join
from keras.models import load_model
from skimage import io
from skimage.transform import resize
from skimage import draw
from classifier import create_model
import numpy as np
from edge_finder import calc_bounds


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
    lex, rex, ley, hey = calc_bounds(frame)

    rr, cc = draw.circle_perimeter(lex + 150, ley + 150, 150)
    frame[rr, cc, 0] =  (255, 0, 0)
    # frame = frame[ley:hey, lex:rex, :]
    for_net = cv2.resize(frame, (150, 150))
    for_net = np.swapaxes(for_net, 0, 2)
    for_net = for_net.reshape((1, 3, 150, 150) + for_net.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

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
