from imutils import face_utils
import dlib
import numpy as np
import math
import cv2
import copy

width = 268
height = 268
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./dlib/shape_predictor_68_face_landmarks.dat')


def findFace(img):
    rects = detector(img, 1)
    xMax = 0
    yMax = 0
    wMax = 0
    hMax = 0
    for (i, rect) in enumerate(rects):
        shape = predictor(img, rect)
        shape = face_utils.shape_to_np(shape)

        (x, y, w, h) = cropping(shape)
        if w > wMax:
            (xMax, yMax, wMax, hMax) = (x, y, w, h)
    return (xMax, yMax, wMax, hMax)
    # if wMax == 0:
    #     return img
    # return cropImage(img, xMax, yMax, wMax, hMax, width, height)


def cropping(shape):
    xMin = min(x[0] for x in shape)
    xMax = max(x[0] for x in shape)
    yMin = min(x[1] for x in shape)
    yMax = max(x[1] for x in shape)

    w = xMax - xMin + 10
    h = yMax - yMin + 10
    x = xMin - 5
    y = yMin - 5

    return x, y, w, h


def cropImage(img, x, y, w, h, width, height):
    img = img[y:y + h, x:x + w]
    img = cv2.resize(img, (width, height))
    return img
