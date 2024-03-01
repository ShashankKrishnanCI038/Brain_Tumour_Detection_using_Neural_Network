import cv2
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

image_directory = 'datasets/'

no_tumour_images = os.listdir(image_directory + 'no/')
yes_tumour_images = os.listdir(image_directory + 'yes/')

dataset = []
label = []
INPUTSIZE = 256

for i, image_name in enumerate(no_tumour_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'no/' + image_name)

        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUTSIZE, INPUTSIZE))

        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_tumour_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'yes/' + image_name)

        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUTSIZE, INPUTSIZE))

        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

xtrain, xtest, ytrain, ytest = train_test_split(dataset, label, test_size=0.2, random_state=0)

print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

xtrain = normalize(xtrain, axis=1)
xtest = normalize(xtest, axis=1)

#Basic CNN Model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (INPUTSIZE, INPUTSIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))


model.add(Flatten())

model.add(Dense(256)) #corresponding to INPUTSIZE
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss= 'binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(xtrain, ytrain,
          batch_size=16,
          verbose=1,
          epochs=3,
          validation_data=(xtest, ytest),
          shuffle=False)

test_loss, test_acc = model.evaluate(xtest, ytest)
print(f"This Model error is: {round((test_loss)*100, 2)}% error")
print(f"This Model scores: {round((test_acc)*100, 2)}% accuracy")

model.save('brain_tumour_7epochs.h5')

model = load_model('brain_tumour_7epochs.h5')

image = cv2.imread(r"C:\Users\Vinayaka\Desktop\App Revised\Datasets\pred\pred48.jpg")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

resize = tf.image.resize(image, (INPUTSIZE, INPUTSIZE))
plt.imshow(resize.numpy().astype(int))
plt.show()

pred = model.predict(np.expand_dims(resize/255, axis = 0)).round(2)

if pred[0] >= 0.5:
    print("tumour")
else:
    print("Normal")

logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(xtrain, ytrain,
          batch_size=16,
          verbose=1,
          epochs=3,
          validation_data=(xtest, ytest),
          shuffle=False)

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')

fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')

fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()
