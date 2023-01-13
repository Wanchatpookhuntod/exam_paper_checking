import cv2
import numpy as np
import imutils
import pandas as pd
import time

answers = pd.read_excel('answerExam.xlsx')
cap = cv2.VideoCapture("http://10.50.44.49:8080/video")
# cap = cv2.VideoCapture(0)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def roiExam(im, roi):
    A1 = [110, 130]
    B1 = [150, 170]
    C1 = [190, 210]
    D1 = [230, 250]

    A2 = [384, 404]
    B2 = [424, 444]
    C2 = [464, 484]
    D2 = [504, 524]

    covY = lambda j: (j * 25 - 23) + 175
    covH = lambda j: (j * 25 - 2) + 175

    print(answers)

    for j, choose in enumerate(answers['answers'], start=1):

        chooseIndex = ''

        if j <= 25:
            if choose == "a":
                chooseIndex = A1
            elif choose == "b":
                chooseIndex = B1
            elif choose == "c":
                chooseIndex = C1
            elif choose == "d":
                chooseIndex = D1
            else:
                print(f'Numbers {j} answers not in a b c d')
        else:
            j = j - 25
            if choose == "a":
                chooseIndex = A2
            elif choose == "b":
                chooseIndex = B2
            elif choose == "c":
                chooseIndex = C2
            elif choose == "d":
                chooseIndex = D2
            else:
                print(f'Numbers {j} answers not in a b c d')

        numY = covY(j)
        numH = covH(j)

        im[numY: numH, chooseIndex[0]: chooseIndex[1]] = \
            roi[numY: numH, chooseIndex[0]: chooseIndex[1]]
    return im


# cap = cv2.VideoCapture(0)

def warpSheet(cnts, orig, ratio):
    global answersSheet
    for index, c in enumerate(cnts):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        if len(approx) == 4 and index == 0:
            cv2.drawContours(im, [approx], -1, (0, 255, 0), 2)
            warped = four_point_transform(orig, approx.reshape(4, 2) * ratio)
            answersSheet = cv2.resize(warped, (595, 842))
            return answersSheet

click = 0
while True:
    im = cap.read()[1]
    cv2.imshow("out2", im)

    ratio = im.shape[0] / 800.0
    orig = im.copy()
    im = imutils.resize(im, height=800)

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:7]
    answersSheet = warpSheet(cnts, orig, ratio)

    if answersSheet is not None:
        print("OK ====== ")

    key = cv2.waitKey(1)

    if key == ord('c'):
        cv2.putText(im, "Check", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)

        cv2.imread(f"t_image{click}.jpg")
        ratio = im.shape[0] / 800.0
        orig = im.copy()
        im = imutils.resize(im, height=800)

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 75, 200)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:7]
        answersSheet = warpSheet(cnts, orig, ratio)

        try:
            # print(answersSheet)
            answerRoi = cv2.cvtColor(answersSheet, cv2.COLOR_BGR2GRAY)
            formSheetResult = np.ones((answersSheet.shape[0], answersSheet.shape[1]), dtype=np.uint8) * 255
            sheetResult = roiExam(formSheetResult, answerRoi)

            resultBlur = cv2.blur(sheetResult, (1, 1), 5)
            _, resultThresh = cv2.threshold(resultBlur, 100, 225, cv2.THRESH_BINARY_INV)

            kernel = np.ones((5, 5), np.uint8)
            resultDilate = cv2.dilate(resultThresh, kernel, iterations=1)

            resultEdged = cv2.Canny(resultDilate, 75, 200)
            contoursResult = cv2.findContours(resultEdged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

            score = len(contoursResult)

            # draw process
            font = cv2.FONT_HERSHEY_SIMPLEX
            for m, c in enumerate(contoursResult, start=1):
                (x, y), r = cv2.minEnclosingCircle(c)
                cv2.circle(answersSheet, (int(x), int(y)), int(10), (0, 0, 255), 1)
                cv2.putText(answersSheet, f"{((m - score) * -1) + 1}",
                            (260 if x < 300 else 539, int(y)), font, .5, (0, 0, 255), 1)

            cv2.putText(answersSheet, f"score: {score}",
                        (answersSheet.shape[1] - 120, 80), font, .7, (0, 0, 255), 1)

            (w, h), b = cv2.getTextSize(f"score: {score}", font, .7, 1)

            cv2.rectangle(answersSheet, (answersSheet.shape[1] - 120 - 10, 80 + 10),
                          (answersSheet.shape[1] - 120 + w + 10, 80 - h - 10), (0, 0, 255), 1)

            # print(score)

            cv2.imshow('out', answersSheet)

        except:
            pass



    elif key == ord("w"):
        click += 1
        cv2.putText(im,"Capture", (20, 60),cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,0), 2)
        # cv2.imwrite(f"{time.strftime('%H_%M_%S')}.jpg", im)
        cv2.imwrite(f"t_image{click}.jpg", im)

    elif key == 27:
        break

    else:
        pass

cap.release()
cv2.destroyAllWindows()