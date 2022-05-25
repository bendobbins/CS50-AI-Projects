import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
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
    model.evaluate(x_test,  y_test, verbose=2)

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
    category = 0
    images = []
    labels = []
    dimensions = (IMG_WIDTH, IMG_HEIGHT)
    # Path to user's current directory + directory of directories
    dataPath = os.path.join(os.getcwd(), data_dir)

    while category < NUM_CATEGORIES:
        # Add path to directory with images
        catDataPath = os.path.join(dataPath, str(category))

        for image in os.listdir(catDataPath):
            # For each image in the path, convert it into a numpy array, resize
            imgArrayOrig = cv2.imread(os.path.join(catDataPath, image))
            imgArray = cv2.resize(imgArrayOrig, dimensions)
            # Add resized image to images and label for image to labels
            images.append(imgArray)
            labels.append(category)
            print(f"Image {image} from category {category} successfully converted to numpy array.")
        category += 1
    
    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([

        # Convolutional layer with 32 filters and 3x3 kernel
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(30, 30, 3)
        ),

        # Max-pooling with 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Repeat both steps above
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(30, 30, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten
        tf.keras.layers.Flatten(),

        # First hidden layer with 0.25 dropout
        tf.keras.layers.Dense(320, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        # Second hidden layer
        tf.keras.layers.Dense(70, activation="relu"),
        tf.keras.layers.Dropout(0.03),

        # Output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    
    # Compile model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
