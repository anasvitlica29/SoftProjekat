import cv2
import detectFace
import tryGender

cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (x, y, w, h) = detectFace.findFace(gray)  # dlib in action
    roi_gray = gray[y:y + h, x:x + w]  # region of interest
    # img_item = "my_image.png"
    # cv2.imwrite(img_item, roi_gray)

    color = (255, 0, 0)
    stroke = 2
    end_coordx = x + w
    end_coordy = y + h
    cv2.rectangle(frame, (x, y), (end_coordx, end_coordy), color, stroke)

    #Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    # tekst = tryGender.recognize(roi_gray)
    color = (255, 255, 255)
    stroke = 2
    cv2.putText(frame, "tekst", (x, y), font, 1, color, stroke, cv2.LINE_AA)


    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
