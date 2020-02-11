import os
from sklearn.neighbors import KNeighborsClassifier
from joblib.numpy_pickle_utils import xrange
import cv2
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD
import pickle
import dlib
import numpy as np


def findFace(frame, detector, predictor):
    try:
        dets = detector(frame, 1)
        ret = np.matrix([[p.x, p.y] for p in predictor(frame, dets[0]).parts()])
        return ret
    except:
        pass


def euclidean_distance(pozicija1, pozicija2):
    x = (pozicija1[0] - pozicija2[0]) * (pozicija1[0] - pozicija2[0])
    y = (pozicija1[1] - pozicija2[1]) * (pozicija1[1] - pozicija2[1])
    res = np.sqrt(x + y)
    return res


def proportions(img, predictor, detector):
    X = []
    dots = findFace(img, detector, predictor)

    try:
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

        X.append(leftEyebrow)
        X.append(rightEyebrow)
        X.append(glabella)
        X.append(chin)
        X.append(leftNostril)
        X.append(rightNostril)
        X.append(nose)
        X.append(leftOuterEye)
        X.append(leftInnerEye)
        X.append(leftBottomLidInnerEye)
        X.append(leftBottomLidOuterEye)
        X.append(rightInnerEye)
        X.append(rightOuterEye)
        X.append(rightBottomLidInnerEye)
        X.append(rightBottomLidOuterEye)
        X.append(mouthLeft)
        X.append(mouthRight)
        X.append(mouthUpperLipMiddle)
        X.append(mouthLowerLipMiddle)
        X.append(mouthUpperLipLeft)
        X.append(mouthUpperLipRight)
        features = np.asarray(X)

        faceHeight = euclidean_distance(glabella, chin)

        proportions = []
        for i in xrange(0, features.shape[0] - 1):
            for j in xrange(i + 1, features.shape[0]):
                x = features[i]
                y = features[j]
                proportions.append(faceHeight / euclidean_distance(x, y))
        return proportions
    except:
        return None


predictor_path = "dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

ulazM = []
izlazM = []
for ind in xrange(1, 10):
    putanja = "Dataset_Recognition/s1/" + str(ind) + ".jpg"
    slika = cv2.imread(putanja)
    rastojanja = proportions(slika, predictor, detector)
    if rastojanja is None:
        continue
    ulazM.append(rastojanja)
    izlazM.append([1, 0, 0, 0])

for ind in xrange(1, 10):
    putanja = "Dataset_Recognition/s2/" + str(ind) + ".jpg"
    slika = cv2.imread(putanja)
    rastojanja = proportions(slika, predictor, detector)
    if rastojanja is None:
        continue
    ulazM.append(rastojanja)
    izlazM.append([0, 1, 0, 0])

for ind in xrange(1, 10):
    putanja = "Dataset_Recognition/s3/" + str(ind) + ".jpg"
    slika = cv2.imread(putanja)
    rastojanja = proportions(slika, predictor, detector)
    if rastojanja is None:
        continue
    ulazM.append(rastojanja)
    izlazM.append([0, 0, 1, 0])

for ind in xrange(1, 10):
    putanja = "Dataset_Recognition/s4/" + str(ind) + ".jpg"
    slika = cv2.imread(putanja)
    rastojanja = proportions(slika, predictor, detector)
    if rastojanja is None:
        continue
    ulazM.append(rastojanja)
    izlazM.append([0, 0, 0, 1])

ulazNiz = np.asarray(ulazM)
izlazNiz = np.asarray(izlazM)

model = Sequential()
model.add(Dense(11, input_dim=210))
model.add(Activation('sigmoid'))
model.add(Dense(4))
model.add(Activation('sigmoid'))
sgd = SGD(lr=0.1, decay=0.001, momentum=0.7)
model.compile(loss='mean_squared_error', optimizer=sgd)
training = model.fit(ulazNiz, izlazNiz, nb_epoch=2000, batch_size=400, verbose=1)

with open('mrezaRec', 'wb') as f:
    pickle.dump(model, f)