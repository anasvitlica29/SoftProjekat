import os
import functions
import cv2
import tryGender


train_dir_women = './Dataset_Pol/zene'
train_dir_men = './Dataset_Pol/muskarci'

img_path = os.path.join(train_dir_men, '50.png')
img = functions.load_image(img_path)

# cv2.imshow('Naziv', img)
# cv2.waitKey()

#test sa web cam
camera = cv2.VideoCapture(0)
for i in range(5):
    return_value, image = camera.read()
    cv2.imwrite('opencv' + str(i) + '.png', image)
del camera

img_path = os.path.join('./', 'opencv2.png')
img = functions.load_image(img_path)

# cv2.imshow('ja', img)
# cv2.waitKey()
tekst = tryGender.recognize(img)
print(tekst)





