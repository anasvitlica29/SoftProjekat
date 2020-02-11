import cv2
import siameseNew

img1_path = 'Dataset_Recognition/s1/7.jpg';
img1 = cv2.imread(img1_path)

img_test_path = 'IMG_TEST/marija/1.jpg'
img_test = cv2.imread(img_test_path)

model = siameseNew.load_model()  # ValueError: Unknown loss function:contrastive_loss
# model.predict([])
