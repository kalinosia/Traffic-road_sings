import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from tensorflow import keras
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

"""
Understanding and help
- Documentation
- https://cs50.harvard.edu/ai/2020/notes/5/
- https://cs50.harvard.edu/ai/2020/projects/5/traffic/
- https://www.youtube.com/watch?v=u7n9t1cBei8&list=PLWKjhJtqVAblStefaz_YOVpDWqcRScc2s&index=15
- https://www.youtube.com/watch?v=6g4O5UOH304&list=PLWKjhJtqVAblStefaz_YOVpDWqcRScc2s&index=11
- https://www.machinecurve.com/index.php/2020/03/30/how-to-use-conv2d-with-keras/
- https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/ (!)
"""


EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43  # 3 # when gtrsb-small
TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    # print("Loading date.......")

    images = []
    labels = []

    for i in range(0, NUM_CATEGORIES):
        path = (os.path.join(data_dir, str(i)))

        img_files_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        for img in img_files_names:
            image = cv2.imread(os.path.join(path, img))

            # resize image
            """[optional] flag that takes one of the following methods. 
            INTER_NEAREST – a nearest-neighbor interpolation 
            INTER_LINEAR – a bilinear interpolation (used by default) 
            INTER_AREA – resampling using pixel area relation. 
            It may be a preferred method for image decimation, as it gives moire’-free results. 
            But when the image is zoomed, it is similar to the INTER_NEAREST method. 
            INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood 
            INTER_LANCZOS4 – a Lanczos interpolation over 8×8 pixel neighborhood"""
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            # BGR -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # if not there will be problem with color
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # in gray scale to much lost..

            images.append(image)
            labels.append(i)
            # print(image.shape) # (30,30,3) IT'S OKAY

    # print("LEN OF IMAGES: ", len(images))  # small 840 # all

    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    ~`and 3 values for each pixel for red, green, and blue).......
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # print("Doing model...")

    model = tf.keras.Sequential([

        # Zamienia [[[..],..[..]]..] na [......]
        # keras.layers.Flatten(input_shape=(IMG_WIDTH,IMG_HEIGHT,3)),
        # Wartości kanałów RGB w [0, 255] zakresu.Tutaj standaryzacji wartości będzie w [0, 1]
        keras.layers.experimental.preprocessing.Rescaling(1. / 255),

        keras.layers.Conv2D(32, 3, padding="same", activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT,3)),
        keras.layers.MaxPooling2D(),  # Max pooling is then used to reduce the spatial dimensions
        keras.layers.Conv2D(64, 3, padding="same", activation='relu'),  # We then learn 64 filters
        keras.layers.MaxPooling2D(),  # pool_size=(2, 2)?????
        keras.layers.Conv2D(128, 3, padding="same", activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        # Hidden layer with 128 neurons, Dense - fully conected
        keras.layers.Dense(128, activation='relu'),
        # output
        keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    # model.add(Conv2D(32, (3, 3), activation="relu")) = model.add(Conv2D(32, (3, 3))),model.add(Activation("relu"))

    # Train neural network
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # not sparse_categorical_crossentropy bc 1D
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":
    main()
