import pickle
import cv2
from joblib.numpy_pickle_utils import xrange

import prepoznavanjeLica
import dlib
import numpy as np

with open('mrezaRec', 'rb') as f:
    model = pickle.load(f)


def distance(a, b):
    # Euklidska udaljenost
    x = (a[0] - b[0]) * (a[0] - b[0])
    y = (a[1] - b[1]) * (a[1] - b[1])
    result = np.sqrt(x + y)
    return result


def recognize(img):
    # load dlib detector and predictor
    predictor_path = "dlib/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    new = []
    inp = []

    face = img
    dots = prepoznavanjeLica.prepoznajLice(face, detector, predictor)  # Extracts 68 dots
    leftEyebrow = (dots[19, 0], dots[19, 1])
    rightEyebrow = (dots[24, 0], dots[24, 1])
    glabella = ((leftEyebrow[0] + rightEyebrow[0]) / 2, (leftEyebrow[1] + rightEyebrow[1]) / 2)
    chin = (dots[8, 0], dots[8, 1])
    leftNostril = (dots[31, 0], dots[31, 1])
    rightNostril = (dots[35, 0], dots[35, 1])
    nose = (dots[33, 0], dots[33, 1])
    leftOuterEye = (dots[36, 0], dots[36, 1])
    leftInnerEye = (dots[39, 0], dots[39, 1])
    leftBottomLidInnerEye = (dots[40, 0], dots[40, 1])
    leftBottomLidOuterEye = (dots[41, 0], dots[41, 1])
    rightInnerEye = (dots[42, 0], dots[42, 1])
    rightOuterEye = (dots[45, 0], dots[45, 1])
    rightBottomLidInnerEye = (dots[47, 0], dots[47, 1])
    rightBottomLidOuterEye = (dots[46, 0], dots[46, 1])
    mouthLeft = (dots[48, 0], dots[48, 1])
    mouthRight = (dots[54, 0], dots[54, 1])
    mouthUpperLipMiddle = (dots[51, 0], dots[51, 1])
    mouthLowerLipMiddle = (dots[57, 0], dots[57, 1])
    mouthUpperLipLeft = (dots[50, 0], dots[50, 1])
    mouthUpperLipRight = (dots[52, 0], dots[52, 1])

    faceHeight = distance(glabella, chin)

    inp.append(leftEyebrow)
    inp.append(rightEyebrow)
    inp.append(glabella)
    inp.append(chin)
    inp.append(leftNostril)
    inp.append(rightNostril)
    inp.append(nose)
    inp.append(leftOuterEye)
    inp.append(leftInnerEye)
    inp.append(leftBottomLidInnerEye)
    inp.append(leftBottomLidOuterEye)
    inp.append(rightInnerEye)
    inp.append(rightOuterEye)
    inp.append(rightBottomLidInnerEye)
    inp.append(rightBottomLidOuterEye)
    inp.append(mouthLeft)
    inp.append(mouthRight)
    inp.append(mouthUpperLipMiddle)
    inp.append(mouthLowerLipMiddle)
    inp.append(mouthUpperLipLeft)
    inp.append(mouthUpperLipRight)
    features = np.asarray(inp)

    props = []
    for i in xrange(0, features.shape[0] - 1):
        for j in xrange(i + 1, features.shape[0]):
            x = features[i]
            y = features[j]
            props.append(faceHeight / distance(x, y))
    new.append(props)
    X = np.asarray(new)
    t = model.predict(X)

    print(t)
    ret = ''
    if t[0][0] > t[0][1] and t[0][0] > t[0][2] and t[0][0] > t[0][3]:
        ret = "Marija"
    elif t[0][1] > t[0][0] and t[0][1] > t[0][2] and t[0][1] > t[0][3]:
        ret = "Mihailo"
    elif t[0][2] > t[0][0] and t[0][2] > t[0][1] and t[0][2] > t[0][3]:
        ret = "Zoran"
    elif t[0][3] > t[0][0] and t[0][3] > t[0][1] and t[0][3] > t[0][2]:
        ret = "Ana"

    return ret

def recognize1(img):
    # load dlib detector and predictor
    predictor_path = "dlib/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    new = []
    inp = []

    face = img
    dots = prepoznavanjeLica.prepoznajLice(face, detector, predictor)  # Extracts 68 dots
    widthLeft = (dots[0, 0], dots[1, 0])
    widthRight = (dots[16, 0], dots[16, 1])
    faceWidth = distance(widthRight, widthLeft)

    leftEyeInnerCorner = (dots[39, 0], dots[39, 1])
    rightEyeInnerCorner = (dots[42, 0], dots[42, 0])
    eyeDistance = faceWidth / distance(leftEyeInnerCorner, rightEyeInnerCorner)

    inp.append(eyeDistance)

    leftEyeOuterCorner = (dots[36, 0], dots[36, 1])
    rightEyeOuterCorner = (dots[45, 0], dots[45, 1])
    leftEyeWidth = faceWidth / distance(leftEyeInnerCorner, leftEyeOuterCorner)
    rightEyeWidth = faceWidth / distance(rightEyeInnerCorner, rightEyeOuterCorner)

    inp.append(leftEyeWidth)
    inp.append(rightEyeWidth)

    leftEyebrow = (dots[19, 0], dots[19, 1])
    rightEyebrow = (dots[24, 0], dots[24, 1])
    glabella = ((leftEyebrow[0] + rightEyebrow[0]) / 2, (leftEyebrow[1] + rightEyebrow[1]) / 2)
    chin = (dots[8, 0], dots[8, 1])
    faceHeight = distance(glabella, chin)

    noseTop = (dots[27, 0], dots[27, 1])
    noseBottom = (dots[33, 0], dots[33, 1])
    noseLength = faceHeight / distance(noseTop, noseBottom)
    noseLeft = (dots[31, 0], dots[31, 1])
    noseRight = (dots[35, 0], dots[35, 1])
    noseWidth = faceWidth / distance(noseLeft, noseRight)

    inp.append(noseLength)
    inp.append(noseWidth)

    topLipMiddle = (dots[51, 0], dots[51, 1])
    distanceNoseMouth = faceHeight / distance(noseBottom, topLipMiddle)

    inp.append(distanceNoseMouth)

    lipsLeftCorner = (dots[48, 0], dots[48, 1])
    lipsRightCorner = (dots[54, 0], dots[54, 1])
    lipsWidth = faceWidth / distance(lipsLeftCorner, lipsRightCorner)

    inp.append(lipsWidth)

    diagonalLeft = faceHeight / distance(widthLeft, chin)
    diagonalRight = faceHeight / distance(widthRight, chin)

    inp.append(diagonalLeft)
    inp.append(diagonalRight)

    features = np.asarray(inp)

    # props = []
    # for i in xrange(0, features.shape[0] - 1):
    #     for j in xrange(i + 1, features.shape[0]):
    #         x = features[i]
    #         y = features[j]
    #         props.append(faceHeight / distance(x, y))
    new.append(features)
    X = np.asarray(new)
    t = model.predict(X)

    print(t)
    ret = ''
    if t[0][0] > t[0][1] and t[0][0] > t[0][2] and t[0][0] > t[0][3]:
        ret = "Marija"
    elif t[0][1] > t[0][0] and t[0][1] > t[0][2] and t[0][1] > t[0][3]:
        ret = "Mihailo"
    elif t[0][2] > t[0][0] and t[0][2] > t[0][1] and t[0][2] > t[0][3]:
        ret = "Zoran"
    elif t[0][3] > t[0][0] and t[0][3] > t[0][1] and t[0][3] > t[0][2]:
        ret = "Ana"

    return ret


# img_path = "IMG_TEST/marija/9.jpg"
# img = cv2.imread(img_path)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# res = recognize(gray)
# print(res)
