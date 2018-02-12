from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import imutils
import numpy as np
import cv2


def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow("trackbar", cv2.WINDOW_NORMAL)

# easy assigments
hl = "Hue Low"
hh = "Hue High"
sl = "Saturation Low"
sh = "Saturation High"
vl = "Value Low"
vh = "Value High"

cv2.createTrackbar(hl, "trackbar", 0, 255, nothing)
cv2.createTrackbar(hh, "trackbar", 0, 255, nothing)
cv2.createTrackbar(sl, "trackbar", 0, 255, nothing)
cv2.createTrackbar(sh, "trackbar", 0, 255, nothing)
cv2.createTrackbar(vl, "trackbar", 0, 255, nothing)
cv2.createTrackbar(vh, "trackbar", 0, 255, nothing)

while (1):
    _, frame = cap.read()
    # import the necessary packages


    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    # convert to HSV from BGR
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # read trackbar positions for all
    hul = cv2.getTrackbarPos(hl, "trackbar")
    huh = cv2.getTrackbarPos(hh, "trackbar")
    sal = cv2.getTrackbarPos(sl, "trackbar")
    sah = cv2.getTrackbarPos(sh, "trackbar")
    val = cv2.getTrackbarPos(vl, "trackbar")
    vah = cv2.getTrackbarPos(vh, "trackbar")
    # make array for final values
    HSVLOW = np.array([hul, sal, val])
    HSVHIGH = np.array([huh, sah, vah])

    # apply the range on a mask
    mask = cv2.inRange(hsv, HSVLOW, HSVHIGH)
    bitwise_and = cv2.bitwise_and(frame, frame, mask=mask)
    bitwise_not = cv2.bitwise_not(frame, mask=mask)
    bitwise_or = cv2.bitwise_or(frame, frame, mask=mask)
    bitwise_xor = cv2.bitwise_xor(frame, frame, mask=mask)

    cv2.imshow("frame", frame)
    cv2.imshow("bitwise_and", bitwise_and)
    cv2.imshow("bitwise_not", bitwise_not)
    cv2.imshow("bitwise_or", bitwise_or)
    cv2.imshow("bitwise_xor", bitwise_xor)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
