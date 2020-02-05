import dlib
import numpy as np
import cv2

def prepoznajLice(frame,detector, predictor):
    try:
        # type: (object, object, object) -> object
        dets = detector(frame,1)
        matrica = np.matrix([[p.x, p.y] for p in predictor(frame, dets[0]).parts()])
       # for i in xrange(0, 27):
            #pozicija = (matrica[ i, 0], matrica[ i, 1])
            #cv2.circle(frame, pozicija, 5, (0, 0, 255), 1)


        return matrica
    except:
        pass
