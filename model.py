
# coding: utf-8

# In[1]:


import cv2
import csv
import imageio
import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Flatten, Dense, Convolution2D, Lambda, BatchNormalization, Dropout
from keras import backend as K
from sklearn.utils import shuffle as sk_shuffle
from sklearn.model_selection import train_test_split
from numba import cuda


# In[10]:


def load_samples(data_path,
                 csv_file_name='driving_log.csv',
                 samples=[],
                 correction_left=0.23,
                 correction_right=0.27,
                 drop_prob=0.7):
    """
    Loads all the samples in the a single list. Does not load the image, but sets the correct path
    so they can be loaded by the generators.

    Args:
        data_path: str: The path to the data directory with the slash at the end
        csv_file_name: str: The name of csv_file that should be used.
        samples: list: The list of samples, starts with empty one.
        correction_left: float: The correction that should be used for the left image
        correct_right: float: The correction that should be applied for the right image

    Returns:
        samples: list: The list of samples with only one image path and the value of steering
    """
    csv_file_path = str(data_path) + str(csv_file_name)
    image_path = str(data_path) + "IMG/"
    with open(csv_file_path) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            for i in range(3):
                file_name = get_file_name(line[i])
                file_path = str(image_path) + str(file_name)
                basic_steering = float(line[3])
                # Drop the streing that is negative or zero with probability of 0.7
                if basic_steering == 0 and np.random.rand() < drop_prob:
                    continue
                data = [file_path]
                if i == 0:
                    data.append(basic_steering)
                if i == 1:
                    data.append(basic_steering + correction_left)
                if i == 2:
                    data.append(basic_steering - correction_right)
                samples.append(data)
    return samples


def read_image(image_path):
    """
    Reads the image to the given array

    Args:
        image_path: str: The path to the image file that should be read

    Returns:
        image: list: The array like image or PIL image

    """
    return imageio.imread(image_path)


def flip_image(image, basic_steering):
    """
    Flips the image

    Args:
        image: list: The array like image or PIL image
        basic_steering: float: The basic steering value

    Returns:
        flipped_image: list: The array like image or PIL image
        basic_steering_flipped: float: The basic steering value for flipped image
    """
    image_flipped = np.fliplr(image)
    basic_steering_flipped = - basic_steering
    return image_flipped, basic_steering_flipped


def get_file_name(file_field=""):
    """
    Returns the file name for the given field

    Args:
        file_field: str: The file field for the given file

    Returns:
        str: The name of the file in the field
    """
    return file_field.split('/')[-1]


def generator(samples, batch_size=32, with_flipped=True):
    """
    Generates the needed images in the given batch size

    Args:
        samples: list: The complete list of samples that should be used
        batch_size: int: The size of batches that should be used
        with_flipped: bool: Should the flipped images also be used.

    Yields:

        X_train, the images for the training
        y_train, the labels for the training
    """
    num_samples = len(samples)
    # Always
    while 1:
        sk_shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                # Add the main image
                image = read_image(batch_sample[0])
                measurement = batch_sample[1]
                images.append(image)
                measurements.append(measurement)
                # Flipped image
                if with_flipped:
                    flipped, steering = flip_image(image, measurement)
                    images.append(flipped)
                    measurements.append(steering)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sk_shuffle(X_train, y_train)


# In[11]:


# Load the data as samples
# data_2 is my own recorded data
# data is the provided data from udacity

samples = load_samples('data_2/', samples=[])
samples = load_samples('data/', samples=samples)
print('All Samples: ', len(samples))


# In[12]:


# Create the needed sets
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print('Training Samples:', len(train_samples))
print('Validation Samples: ', len(validation_samples))


# In[13]:


# Create the generators
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# In[14]:


# Nvidia Model
def create_training_model(drop_prob=0.5, input_shape=(160, 320, 3), learning_rate=0.0009):
    """
    Creates the training with help of Keras

    Args:
        drop_prob: float: The dropout probability
        input_shape: tuple: The shape of the input image, defaults to (160,320,3)
        learning_rate: float: The value of training_loss that is 0.0009

    Return:
        model: The sequential model that is created with keras
    """
    model = Sequential()
    model.add(Lambda(lambda x: x - 255.0 / 255.0, input_shape=input_shape))
    model.add(BatchNormalization(axis=3))
    model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization(axis=3))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(drop_prob))
    model.add(Dense(50, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(drop_prob))
    model.add(Dense(10, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    return model


# In[15]:


# Add the model data and fitting
model = create_training_model()
model.summary()


# In[16]:


batch_size = 32
model.fit_generator(train_generator,
                    steps_per_epoch=int(len(train_samples)/batch_size),
                    validation_data=validation_generator,
                    validation_steps=int(len(validation_samples)/batch_size),
                    epochs=7)
model.save('nvidia_model_2.h5')


# In[17]:


# Empty the memory of GPU
K.clear_session()
cuda.select_device(0)
cuda.close()

