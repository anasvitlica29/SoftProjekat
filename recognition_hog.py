import numpy as np
import cv2  # OpenCV
from sklearn.svm import SVC  # SVM klasifikator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # KNN
from joblib import dump, load
import matplotlib
import matplotlib.pyplot as plt
from joblib.numpy_pickle_utils import xrange
import os


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)


def display_image(image):
    plt.imshow(image, 'gray')
    plt.show()


def compute_hog(img):
    nbins = 9  # broj binova
    cell_size = (8, 8)  # broj piksela po celiji
    block_size = (3, 3)  # broj celija po bloku

    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
    return hog


def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx * ny))


s1 = []
s2 = []
s3 = []
s4 = []
labels = []
for ind in xrange(1, 10):
    putanja = "Dataset_Recognition/s1/" + str(ind) + ".jpg"
    slika = cv2.imread(putanja)
    hog = compute_hog(slika).compute(slika)
    s1.append(hog)
    labels.append(0)

for ind in xrange(1, 10):
    putanja = "Dataset_Recognition/s2/" + str(ind) + ".jpg"
    slika = cv2.imread(putanja)
    hog = compute_hog(slika).compute(slika)
    s2.append(hog)
    labels.append(1)

for ind in xrange(1, 10):
    putanja = "Dataset_Recognition/s3/" + str(ind) + ".jpg"
    slika = cv2.imread(putanja)
    hog = compute_hog(slika).compute(slika)
    s3.append(hog)
    labels.append(2)

for ind in xrange(1, 10):
    putanja = "Dataset_Recognition/s4/" + str(ind) + ".jpg"
    slika = cv2.imread(putanja)
    hog = compute_hog(slika).compute(slika)
    s4.append(hog)
    labels.append(3)

x = np.vstack((s1, s2, s3, s4))
y = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = reshape_data(x_train)
x_test = reshape_data(x_test)

try:
    # clf_knn = load('knn_recognition.joblib')
    # y_train_pred = clf_knn.predict(x_train)
    # y_test_pred = clf_knn.predict(x_test)
    # print("Train accuracy: ", accuracy_score(y_train, y_train_pred))  # Train accuracy:  0.42857142857142855
    # print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))  # alidation accuracy:  0.5

    clf_svm = load('svm_recognition.joblib')
    y_train_pred = clf_svm.predict(x_train)
    y_test_pred = clf_svm.predict(x_test)

    # print("Train accuracy: ", accuracy_score(y_train, y_train_pred))  # Train accuracy:  1.0
    # print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))  # Validation accuracy:  0.75
except:
    # SVM classifier
    clf_svm = SVC(kernel='linear', probability=True)
    clf_svm = clf_svm.fit(x_train, y_train)
    dump(clf_svm, 'svm_recognition.joblib')
    y_train_pred = clf_svm.predict(x_train)
    y_test_pred = clf_svm.predict(x_test)
    # print("Train accuracy: ", accuracy_score(y_train, y_train_pred))  # Train accuracy:  1.0
    # print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))  # Validation accuracy:  0.75

    # KNN classifier
    # clf_knn = KNeighborsClassifier(n_neighbors=10)
    # clf_knn = clf_knn.fit(x_train, y_train)
    # dump(clf_knn, 'knn_recognition.joblib')
