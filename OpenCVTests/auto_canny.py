# import the necessary packages
import numpy as np
import cv2
import imutils

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def nothing(x):
    pass

cv2.namedWindow("trackbar", cv2.WINDOW_NORMAL)
cv2.createTrackbar("threshold1", "trackbar", 0, 255, nothing)
cv2.createTrackbar("threshold2", "trackbar", 60, 255, nothing)

cap = cv2.VideoCapture(1)

while (1):
    _, image = cap.read()

    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # apply Canny edge detection using a wide threshold, tight
    # threshold, and automatically determined threshold
    wide = cv2.Canny(blurred, 10, 200)
    tight = cv2.Canny(blurred, 225, 250)
    auto = auto_canny(blurred, 0.33)

    # show the images
    cv2.imshow("Original", image)
    cv2.imshow("Edges_Wide", wide)
    cv2.imshow("Edges_Tight", tight)
    cv2.imshow("Edges_Auto", auto)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
