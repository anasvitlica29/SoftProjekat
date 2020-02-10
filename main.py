import os
import functions
import cv2
import tryGender
import detectFace


def try_gender_from_dataset():
    train_dir_women = './Dataset_Pol/zene'
    train_dir_men = './Dataset_Pol/muskarci'

    img_path = os.path.join(train_dir_women, '118.png')
    img = functions.load_image(img_path)

    tekst = tryGender.recognize(img)
    print(tekst)


def prepareImages():
    path = 'C:\Ana PSW\SoftProjekat\IMG_TEST\marija'
    width = 268
    height = 268
    i = 1
    for image_path in os.listdir(path):
        input_path = os.path.join(path, image_path)
        image = cv2.imread(input_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        (x, y, w, h) = detectFace.findFace(gray)  # dlib in action
        print(x)
        roi_gray = gray[y:y + h, x:x + w]  # region of interest
        resized = cv2.resize(roi_gray, (width, height), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite("{}.jpg".format(i), resized)
        i += 1

