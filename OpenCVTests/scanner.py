# import the necessary packages
from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils

def nothing(x):
    pass
cv2.namedWindow("trackbar", cv2.WINDOW_NORMAL)
cv2.createTrackbar("threshold1", "trackbar", 0, 255, nothing)
cv2.createTrackbar("threshold2", "trackbar", 60, 255, nothing)

cap = cv2.VideoCapture(0)

while (1):
    _, frame = cap.read()

    #frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    ratio = frame.shape[0] / 500.0
    orig = frame.copy()
    image = imutils.resize(frame, height = 500)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    threshold1 = cv2.getTrackbarPos("threshold1", "trackbar")
    threshold2 = cv2.getTrackbarPos("threshold2", "trackbar")
    edged = cv2.Canny(gray, threshold1, threshold2)

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    screenCnt = None

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    # show the contour (outline) of the piece of paper
    if screenCnt is not None:
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

        # apply the four point transform to obtain a top-down
        # view of the original image
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

        # convert the warped image to grayscale, then threshold it
        # to give it that 'black and white' paper effect
        #warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        #T = threshold_local(warped, 11, offset = 10, method = "gaussian")
        #warped = (warped > T).astype("uint8") * 255

    # show the original and scanned images
    # show the original image and the edge detected image
    cv2.imshow("Edged", edged)
    cv2.imshow("Outline", image)
    cv2.imshow("Original", imutils.resize(orig, height = 650))
    cv2.imshow("Scanned", imutils.resize(warped, height = 650))

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
