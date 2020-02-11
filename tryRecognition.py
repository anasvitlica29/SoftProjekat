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

    ulazNovi = []
    ulaz = []

    face = img
    tackeLica = prepoznavanjeLica.prepoznajLice(face, detector, predictor)  # Extracts 68 dots
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

    visinaLica = distance(gornjaSrednja, brada)

    ulaz.append(levaObrva)
    ulaz.append(desnaObrva)
    ulaz.append(gornjaSrednja)
    ulaz.append(brada)
    ulaz.append(levaNozdrva)
    ulaz.append(desnaNozdrva)
    ulaz.append(dnoNosa)
    ulaz.append(levoLevoOko)
    ulaz.append(levoDesnoOko)
    ulaz.append(levoDoleDesnoOko)
    ulaz.append(levoDoleLevoOko)
    ulaz.append(desnoLevoOko)
    ulaz.append(desnoDesnoOko)
    ulaz.append(desnoDoleLevoOko)
    ulaz.append(desnoDoleDesnoOko)
    ulaz.append(ustaLeviUgao)
    ulaz.append(ustaDesniugao)
    ulaz.append(ustaSredinaGore)
    ulaz.append(ustaSredinaDole)
    ulaz.append(ustaGoreLevo)
    ulaz.append(ustaGoreDesno)
    svi = np.asarray(ulaz)
    rastojanja = []
    for br in xrange(0, svi.shape[0] - 1):
        for dr in xrange(br + 1, svi.shape[0]):
            pozicija1 = svi[br]
            pozicija2 = svi[dr]
            rastojanja.append(visinaLica / distance(pozicija1, pozicija2))
    ulazNovi.append(rastojanja)
    input1 = np.asarray(ulazNovi)
    t = model.predict(input1)

    print(t)
    ret = ''
    if t[0][0] > t[0][1] and t[0][0] > t[0][2] and t[0][0] > t[0][3]:
        ret = 'Marija'
    elif t[0][1] > t[0][0] and t[0][1] > t[0][2] and t[0][1] > t[0][3]:
        ret = 'Mihailo'
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
