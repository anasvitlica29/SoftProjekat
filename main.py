import os
import functions
import cv2
import tryGender


train_dir_women = './Dataset_Pol/zene'
train_dir_men = './Dataset_Pol/muskarci'

img_path = os.path.join(train_dir_women, '118.png')
img = functions.load_image(img_path)

tekst = tryGender.recognize(img)
print(tekst)





