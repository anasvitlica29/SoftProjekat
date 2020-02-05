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

tekst = tryGender.prepoznaj(img)
print(tekst)





