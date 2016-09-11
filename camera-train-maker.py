import cv2
# import pickle
from os.path import join

PREVIEW = "preview"

OUTPUT_FOLDER = 'data/yellow_potato'

def main():
    cv2.namedWindow(PREVIEW)
    vc = cv2.VideoCapture(0)
    i = 0

    if vc.isOpened():
        rv, rf = vc.read()
    else:
        rv = False

    print("while")
    while rv:
        rv, rf = vc.read()
        rf = do(rf)

        cv2.imshow(PREVIEW, rf)

        key = cv2.waitKey(20)
        if key == 115: # s
            filename = join(OUTPUT_FOLDER, "{}.png".format(i))
            cv2.imwrite(filename, rf)
            print("saved: " + filename)
            i+=1

        if key == 27: # esc
            break
    cv2.destroyWindow(PREVIEW)


def do(rf):
    # rf = cv2.cvtColor(rf, cv2.COLOR_BGR2GRAY)
    # rf = cv2.resize(rf, (96, 96))

    # res = net1.predict(rf.reshape(1,9216) / 255)[0]
    # res = net1.predict(rf.reshape(1,1,96,96) / 255)[0]
    # for x,y in zip(res[0::2]*48+48, res[1::2]*48+48):
    #     cv2.drawMarker(rf, (int(x), int(y)), 255)

    return rf


if __name__ == "__main__":
    main()
