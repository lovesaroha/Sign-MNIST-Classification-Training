# Love Saroha
# lovesaroha1994@gmail.com (email address)
# https://www.lovesaroha.com (website)
# https://github.com/lovesaroha  (github)

# Train keras model on sign MNIST data.
import csv
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parameters.
epochs = 20
batchSize = 128

# Training and validation data url (https://www.kaggle.com/datamunge/sign-language-mnist/version/1).

# Get data from csv file.


def get_data(filename):
    with open(filename) as csvfile:
        csv_reader = csv.reader(
            csvfile, delimiter=',')
        skip = True
        temp_images = []
        temp_labels = []
        for row in csv_reader:
            if skip:
                skip = False
            else:
                temp_labels.append(row[0])
                image_data = row[1:785]
                image_data_as_array = numpy.array_split(image_data, 28)
                temp_images.append(image_data_as_array)
    return numpy.array(temp_images).astype('float'), numpy.array(temp_labels).astype('float')


# Load training images from csv file.
training_images, training_labels = get_data(
    "sign_mnist_train.csv")
training_images = numpy.expand_dims(training_images, axis=3)

# Load validation images from csv file.
validation_images, validation_labels = get_data(
    "sign_mnist_test.csv")
validation_images = numpy.expand_dims(validation_images, axis=3)

# Create model with 26 output units for classification.
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(26, activation='softmax')
])

# Set loss function and optimizer.
model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])


class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
      # Stop when validation accuracy is more than 98%.
        if logs.get('val_accuracy') is not None and logs.get('val_accuracy') > 0.98:
            print("\nTraining Stopped!")
            self.model.stop_training = True


# Callback function to check accuracy.
checkAccuracy = myCallback()

# Train model.
model.fit(ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
).flow(training_images, training_labels, batch_size=batchSize),
    steps_per_epoch=len(training_images) / batchSize,
    epochs=epochs,
    callbacks=[checkAccuracy],
    validation_data=ImageDataGenerator(
    rescale=1. / 255
).flow(validation_images, validation_labels, batch_size=batchSize),
    validation_steps=len(validation_images) / batchSize)

# Predict on a random image.
image = validation_images[6]
prediction = model.predict(image.reshape(1, 28, 28, 1))

# Show image.
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(image)
plt.show()

# Show prediction.
print(prediction)
