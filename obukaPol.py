from joblib.numpy_pickle_utils import xrange
import dlib
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD
import pickle


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


predictor_path = "dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

ulazM = []
izlazM = []
for ind in xrange(1, 139):
    putanja = "Dataset_Pol/muskarci/" + str(ind) + ".png"
    slika = cv2.imread(putanja)
    rastojanja = proporcije(slika, predictor, detector)
    ulazM.append(rastojanja)
    izlazM.append([1, 0])

for ind in xrange(1, 122):
    putanja = "Dataset_Pol/zene/" + str(ind) + ".png"
    slika = cv2.imread(putanja)
    rastojanja = proporcije(slika, predictor, detector)
    ulazM.append(rastojanja)
    izlazM.append([0, 1])

ulazNiz = np.asarray(ulazM)
izlazNiz = np.asarray(izlazM)

model = Sequential()
model.add(Dense(11, input_dim=210))
model.add(Activation('sigmoid'))
model.add(Dense(2))
model.add(Activation('sigmoid'))
sgd = SGD(lr=0.1, decay=0.001, momentum=0.7)
model.compile(loss='mean_squared_error', optimizer=sgd)
training = model.fit(ulazNiz, izlazNiz, nb_epoch=3000, batch_size=400, verbose=1)

with open('mrezaPol', 'wb') as f:
    pickle.dump(model, f)
