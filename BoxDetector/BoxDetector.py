from skimage.filters import threshold_local
import numpy as np
import argparse
import imutils
import cv2

def drawBoundingBox(image, contour, color, thickness):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color, thickness)

    width = int(round(rect[1][1]))
    length = int(round(rect[1][0]))

    return width, length

def calcContourCenter(contour):
    M = cv2.moments(contour)
    cX = int((M["m10"] / M["m00"]))
    cY = int((M["m01"] / M["m00"]))
    return cX, cY

def calcDimension(width_box_px, length_box_px, width_area_px, length_area_px):

    horizRatio = width_area_px / area_width_mm
    vertiRatio = length_area_px / area_length_mm

    width_box_cm = round(width_box_px * horizRatio / 100, 2)
    length_box_cm = round(length_box_px * horizRatio / 100, 2)

    return width_box_cm, length_box_cm

def nothing(x):
    pass

area_width_mm = 210
area_length_mm = 148

cap = cv2.VideoCapture(1)

cv2.namedWindow("trackbar", cv2.WINDOW_NORMAL)
cv2.namedWindow("bifilter", cv2.WINDOW_NORMAL)
cv2.namedWindow("canny", cv2.WINDOW_NORMAL)

# easy assigments
hl = "Hue Low"
hh = "Hue High"
sl = "Saturation Low"
sh = "Saturation High"
vl = "Value Low"
vh = "Value High"

cv2.createTrackbar(hl, "trackbar", 45, 255, nothing)
cv2.createTrackbar(hh, "trackbar", 80, 255, nothing)
cv2.createTrackbar(sl, "trackbar", 0, 255, nothing)
cv2.createTrackbar(sh, "trackbar", 255, 255, nothing)
cv2.createTrackbar(vl, "trackbar", 0, 255, nothing)
cv2.createTrackbar(vh, "trackbar", 255, 255, nothing)

cv2.createTrackbar("d", "bifilter", 5, 10, nothing)
cv2.createTrackbar("sigmaColor", "bifilter", 100, 255, nothing)
cv2.createTrackbar("sigmaSpace", "bifilter", 100, 255, nothing)

cv2.createTrackbar("threshold1", "canny", 50, 255, nothing)
cv2.createTrackbar("threshold2", "canny", 100, 255, nothing)

while (1):
    _, frame = cap.read()

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

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # apply the range on a mask
    mask = cv2.inRange(hsv, HSVLOW, HSVHIGH)
    bitwise_and = cv2.bitwise_and(frame, frame, mask=mask)

    gray = cv2.cvtColor(bitwise_and, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    d = cv2.getTrackbarPos("d", "bifilter")
    sigmaColor = cv2.getTrackbarPos("sigmaColor", "bifilter")
    sigmaSpace = cv2.getTrackbarPos("sigmaSpace", "bifilter")

    blurred = cv2.bilateralFilter(gray, d, sigmaColor, sigmaSpace)
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    kernel = np.ones((6, 6), np.uint8)
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    erode = cv2.erode(dilate, kernel, iterations=2)
    #opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # detect edges
    threshold1 = cv2.getTrackbarPos("threshold1", "canny")
    threshold2 = cv2.getTrackbarPos("threshold2", "canny")
    edged = cv2.Canny(erode, threshold1=50, threshold2=100)

    # blurr canny result to reduce instability while findContours
    edged = cv2.bilateralFilter(edged, 5, 100, 100)

    # find contours
    (_, contours, hierarchy) = cv2.findContours(image=edged, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    # sort contours by areas size descending and keep the top 20
    #contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    # sort contours by areas size descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # detect box parking are - the biggest contour is sure the box parking area
    boxAreaContour = contours.pop(0)
    del contours[0]

    # draw rotated bounding box around parking area and get the dimension in pixel
    width_area_px, length_area_px = drawBoundingBox(frame, boxAreaContour, (255,0,0), 3)

    # detect box contours
    boxContours = []
    for c in contours:
        # filter out unreasonable too small contours
        areaSize = cv2.contourArea(c, oriented=True)
        if areaSize > 500:
            boxContours.append(c)
            # draw bounding rect and get dimension of the box in pixel
            width_box_px, length_box_px = drawBoundingBox(frame, c, (0, 255, 0), 2)

            # compute the dimension in cm
            width_box_cm, length_box_cm = calcDimension(width_box_px, length_box_px, width_area_px, length_area_px)

            # compute the center of the contour
            cX, cY = calcContourCenter(c)
            #cv2.putText(frame, "A:{}".format(areaSize), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(frame, "{0}x{1}".format(width_box_cm, length_box_cm), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # count number of boxes
    numberPackages = len(boxContours)
    cv2.putText(frame, "# Packages: {}".format(numberPackages), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # draw determined contours
    #cv2.drawContours(frame, [boxAreaContour], -1, (255, 0, 0), 1)
    #cv2.drawContours(frame, boxContours, -1, (0, 255, 0), 1)

    cv2.imshow("orig", frame)
    cv2.imshow("bitwise_and", bitwise_and)
    cv2.imshow("thresh", thresh)
    cv2.imshow("erode", erode)
    cv2.imshow("edged", edged)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

exit(0)

