import os
import cv2
import numpy as np
import pickle
import matplotlib as plt
from keras.models import Sequential
from keras.layers import Conv2D, Input
from keras.models import Model
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Lambda, Flatten, Dense
from keras.optimizers import *
from keras import backend as K
import SiameseLoader
import time
import siamese_data_init




def get_siamese_model(input_shape):
    """
        Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape, padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid'))
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    prediction = Dense(1, activation='sigmoid')(L1_distance)
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
    return siamese_net


#################################################

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "Dataset_Recognition")

model = get_siamese_model((105, 105, 1))
model.summary()

optimizer = Adam(lr=0.00006)
model.compile(loss="binary_crossentropy", optimizer=optimizer)

# read_and_label_images
x, y, label_ids = siamese_data_init.read_and_label_images()
# print(x.shape)  # (30, 268, 268, 3) --> 30 slika ukupno, dimenzija 268x268, 3 osobe?
# print(y.shape)  # (30, 1) --> 30 labela

loader = SiameseLoader.get_loader(BASE_DIR)
pairs, targets = loader.make_oneshot_task(20, "train")

evaluate_every = 10  # interval for evaluating on one-shot tasks
loss_every = 20  # interval for printing loss (iterations)
batch_size = 32
n_iter = 2000
N_way = 20  # how many classes for testing one-shot tasks>
n_val = 75  # how many one-shot tasks to validate on?
best = -1
print("Starting training process!")
print("-------------------------------------")
t_start = time.time()
for i in range(1, n_iter):
    (inputs, targets) = loader.get_batch(batch_size)
    loss = model.train_on_batch(inputs, targets)
    print("\n ------------- \n")
    print("Loss: {0}".format(loss))
    if i % evaluate_every == 0:
        print("Time for {0} iterations: {1}".format(i, time.time() - t_start))
        val_acc = loader.test_oneshot(model, N_way, n_val, verbose=True)
        if val_acc >= best:
            print("Current best: {0}, previous best: {1}".format(val_acc, best))
            print("Saving weights to: {0} \n".format(BASE_DIR))
            model.save_weights(BASE_DIR)
            best = val_acc

    if i % loss_every == 0:
        print("iteration {}, training loss: {:.2f},".format(i, loss))

weights_path_2 = os.path.join(BASE_DIR, "model_weights.h5")
model.load_weights(weights_path_2)
