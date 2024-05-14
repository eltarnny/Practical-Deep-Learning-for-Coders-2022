from pathlib import Path

#!pip install tensorflow
# Keras is now preinstalled by TensorFlow
#!pip install --upgrade keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

run_mode = 'train'

current_path = Path.cwd()
data_path = current_path / 'dataset'
training_path = data_path / 'training_set'
test_path = data_path / 'test_set'

# Data Preprocessing

# We do the transformations (image augmentations) to the images to avoid overfitting
# Values as in the Keras example
# rescale is the Feature Scaling
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# target_size is the size that we feed into the CNN
# class_mode binary or categorical
training_set = train_datagen.flow_from_directory(training_path,
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# For the test set, as usual, we do not apply transformations, only Feature Scaling
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Building the CNN

# Init
cnn = tf.keras.models.Sequential()

# Convolution

# filters is the number of Feature Detectors
# Kernel size is the size of the Feature Detector
# activation='relu' , we add the Rectifier Activation Function to reduce linearity
# input_shape: The shape we previously used as target_size and the 3 is for colored images (1 for grayscale images)
cnn.add(tf.keras.layers.Input(shape=(64, 64, 3)))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))

# Pooling

# pool_size the size of the matrix
# By how many pixels we move the matrix
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# 2nd Convolution

# No shape is needed here as it is fed by the previous layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flatten

cnn.add(tf.keras.layers.Flatten())

# Full connection

# As the problem is complex we need many neurons
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Output layer

# As we have only 2 categories, we can use only 1 binary neuron with sigmoid activation function.
# For more categories/neurons, we use softmax activation function
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling

# adam = stochastic
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training (or Loading pre-trained model) for 3 epochs
if run_mode == 'train':
    cnn.fit(x = training_set, validation_data = test_set, epochs = 3)
    # Save for future use
    cnn.save(current_path / 'models' / 'catsanddogsmodel.keras')
else:
    # Load the saved model from the file
    cnn = load_model(current_path / 'models' / 'catsanddogsmodel.keras')
