import os
import pickle
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "Dataset_Recognition")
val_dir = os.path.join(BASE_DIR, "IMG_TEST")


def read_and_label_images():
    current_id = 0
    label_ids = {}

    y_labels = []
    x_train = []

    for root, dirs, files in os.walk(val_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()

                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]
                image = cv2.imread(path)
                image_array = np.array(image)

                x_train.append(image_array)
                y_labels.append(id_)

    # with open("labels.pickle", 'wb') as f:
    #     pickle.dump(label_ids, f)

    y = np.vstack(y_labels)
    x = np.stack(x_train)

    return x, y, label_ids


def read_labels():
    labels = {}
    with open("labels.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}
    return labels


def dump_images_and_label_dict():
    X, y, labels = read_and_label_images()
    with open("train.pickle", "wb") as f:
        pickle.dump((X, labels), f)


# dump_images_and_label_dict()

# X, y, c = read_and_label_images()
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# with open(os.path.join(BASE_DIR, "val.pickle"), "wb") as f:
#     pickle.dump((X, c), f)
