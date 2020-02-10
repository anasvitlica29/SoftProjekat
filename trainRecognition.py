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
import detectFace


# predictor_path = "dlib/shape_predictor_68_face_landmarks.dat"
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(predictor_path)
#
#
# def find_face(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     (x, y, w, h) = detectFace.findFace(gray)
#     return gray[y:y + h, x:x + w], (x, y, w, h)
#
#
# def prepare_data(directory_path):
#     dirs = os.listdir(directory_path)
#
#     persons = []
#     labels = []
#
#     for dir in dirs:
#
#         if not dir.startswith("s"):
#             continue
#
#         label = int(dir.replace("s", ""))  # Labela 1, 2, 3,...
#         images = os.listdir(directory_path + "/" + dir)
#
#         for image in images:
#             img_url = directory_path + "/" + dir + "/" + image
#             img = cv2.imread(img_url)
#
#             face, rect = find_face(img)
#
#             if face is not None:
#                 persons.append(face)
#                 labels.append(label)
#
#     return persons, labels
#
#
# def knn(neighbor, traindata, trainlabel, testdata):
#     neigh = KNeighborsClassifier(n_neighbors=neighbor)
#     neigh.fit(traindata, trainlabel)
#     return neigh.predict(testdata)
#
#
# persons, labels = prepare_data("Dataset_Recognition")
#
# model = Sequential()
# model.add(Dense(11, input_dim=210))
# model.add(Activation('sigmoid'))
# model.add(Dense(2))
# model.add(Activation('sigmoid'))
# sgd = SGD(lr=0.1, decay=0.001, momentum=0.7)
# model.compile(loss='mean_squared_error', optimizer=sgd)
# training = model.fit(persons, labels, nb_epoch=2000, batch_size=400, verbose=1)
#
# with open('mrezaRecognition', 'wb') as f:
#     pickle.dump(model, f)



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


def proporcije(slika, predictor, detector):
    X = []
    tackeLica = findFace(slika, detector, predictor)

    try:
        levaObrva = (tackeLica[19, 0], tackeLica[19, 1])
        desnaObrva = (tackeLica[24, 0], tackeLica[24, 1])
        gornjaSrednja = ((levaObrva[0] + desnaObrva[0]) / 2, (levaObrva[1] + desnaObrva[1]) / 2)
        brada = (tackeLica[8, 0], tackeLica[8, 1])
        levaNozdrva = (tackeLica[31, 0], tackeLica[31, 1])
        desnaNozdrva = (tackeLica[35, 0], tackeLica[35, 1])
        dnoNosa = (tackeLica[33, 0], tackeLica[33, 1])
        levoLevoOko = (tackeLica[36, 0], tackeLica[36, 1])
        levoDesnoOko = (tackeLica[39, 0], tackeLica[39, 1])
        levoDoleDesnoOko = (tackeLica[40, 0], tackeLica[40, 1])
        levoDoleLevoOko = (tackeLica[41, 0], tackeLica[41, 1])
        desnoLevoOko = (tackeLica[42, 0], tackeLica[42, 1])
        desnoDesnoOko = (tackeLica[45, 0], tackeLica[45, 1])
        desnoDoleLevoOko = (tackeLica[47, 0], tackeLica[47, 1])
        desnoDoleDesnoOko = (tackeLica[46, 0], tackeLica[46, 1])
        ustaLeviUgao = (tackeLica[48, 0], tackeLica[48, 1])
        ustaDesniugao = (tackeLica[54, 0], tackeLica[54, 1])
        ustaSredinaGore = (tackeLica[51, 0], tackeLica[51, 1])
        ustaSredinaDole = (tackeLica[57, 0], tackeLica[57, 1])
        ustaGoreLevo = (tackeLica[50, 0], tackeLica[50, 1])
        ustaGoreDesno = (tackeLica[52, 0], tackeLica[52, 1])

        visinaLica = euclidean_distance(gornjaSrednja, brada)

        X.append(levaObrva)
        X.append(desnaObrva)
        X.append(gornjaSrednja)
        X.append(brada)
        X.append(levaNozdrva)
        X.append(desnaNozdrva)
        X.append(dnoNosa)
        X.append(levoLevoOko)
        X.append(levoDesnoOko)
        X.append(levoDoleDesnoOko)
        X.append(levoDoleLevoOko)
        X.append(desnoLevoOko)
        X.append(desnoDesnoOko)
        X.append(desnoDoleLevoOko)
        X.append(desnoDoleDesnoOko)
        X.append(ustaLeviUgao)
        X.append(ustaDesniugao)
        X.append(ustaSredinaGore)
        X.append(ustaSredinaDole)
        X.append(ustaGoreLevo)
        X.append(ustaGoreDesno)
        svi = np.asarray(X)

        rastojanja = []
        for br in xrange(0, svi.shape[0] - 1):
            for dr in xrange(br + 1, svi.shape[0]):
                pozicija1 = svi[br]
                pozicija2 = svi[dr]
                rastojanja.append(visinaLica / euclidean_distance(pozicija1, pozicija2))
        return rastojanja
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
    rastojanja = proporcije(slika, predictor, detector)
    if rastojanja is None:
        continue
    ulazM.append(rastojanja)
    izlazM.append([1, 0, 0])

for ind in xrange(1, 10):
    putanja = "Dataset_Recognition/s2/" + str(ind) + ".jpg"
    slika = cv2.imread(putanja)
    rastojanja = proporcije(slika, predictor, detector)
    if rastojanja is None:
        continue
    ulazM.append(rastojanja)
    izlazM.append([0, 1, 0])

for ind in xrange(1, 10):
    putanja = "Dataset_Recognition/s3/" + str(ind) + ".jpg"
    slika = cv2.imread(putanja)
    rastojanja = proporcije(slika, predictor, detector)
    if rastojanja is None:
        continue
    ulazM.append(rastojanja)
    izlazM.append([0, 0, 1])

ulazNiz = np.asarray(ulazM)
izlazNiz = np.asarray(izlazM)

model = Sequential()
model.add(Dense(11, input_dim=210))
model.add(Activation('sigmoid'))
model.add(Dense(3))
model.add(Activation('sigmoid'))
sgd = SGD(lr=0.1, decay=0.001, momentum=0.7)
model.compile(loss='mean_squared_error', optimizer=sgd)
training = model.fit(ulazNiz, izlazNiz, nb_epoch=2000, batch_size=400, verbose=1)

with open('mrezaRec', 'wb') as f:
    pickle.dump(model, f)