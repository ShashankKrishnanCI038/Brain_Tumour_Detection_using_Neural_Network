import cv2
import os
from PIL import Image
import tensorflow as tf
import numpy as np
from numpy import asarray
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Activation, Dropout, Flatten, Dense
from keras.models import load_model
import streamlit as st
import seaborn as sns
import time

########################################################################################################################


st.title("Neural Network Analysis of Brain Tumour Detection")

#  # progress bar code
# bar = st.progress(0)
# for pr in range(100):
#     time.sleep(0.1)
#     bar.progress(pr + 1)
#     bar.color_picker = 'red'

with open("style.css") as stylefile:
    st.markdown(f"<style>{stylefile.read()}</style>", unsafe_allow_html=True)
    st.markdown("""<meta name="keywords" content="Neural, Network, Detection, Neural Network, Analysis, Kidney Stone Detection, Kidney Stone">
                   <meta name="description" content="Neural Network Analysis of Kidney Stone Detection">
                   <meta name="author" content="Shashank K">
                   <meta name="viewport" content="width=device-width, initial-scale=1.0">
                   <link rel="preconnect" href="https://fonts.googleapis.com">
                   <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
                   <link href="https://fonts.googleapis.com/css2?family=Alata&display=swap" rel="stylesheet">
                   <link href="https://fonts.googleapis.com/css2?family=Josefin+Sans:wght@200&display=swap" rel="stylesheet">""",
                unsafe_allow_html=True)

########################################################################################################################
if st.button("Train"):
    st.write("Training Session Started")

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

    # st.write(len(dataset))
    # st.write(len(label))

    dataset = np.array(dataset)
    label = np.array(label)

    xtrain, xtest, ytrain, ytest = train_test_split(dataset, label, test_size=0.2, random_state=0)

    # st.write(xtrain.shape)
    # st.write(xtest.shape)
    # st.write(ytrain.shape)
    # st.write(ytest.shape)

    xtrain = normalize(xtrain, axis=1)
    xtest = normalize(xtest, axis=1)

    # Basic CNN Model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(INPUTSIZE, INPUTSIZE, 3)))
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

    model.add(Dense(256))  # corresponding to INPUTSIZE
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation(
        'sigmoid'))  # """ for Binary classification = 1 dense, sigmoid and for multiple class classification = 2 dense, softmax"""

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    st.write("Model designed. Kindly wait for the model to be trained and updated")

    model.fit(xtrain, ytrain,
              batch_size=16,
              verbose=1,
              epochs=1,
              validation_data=(xtest, ytest),
              shuffle=False)

    test_loss, test_acc = model.evaluate(xtest, ytest)
    st.write(f"This Model error is: {round(test_loss * 100, 2)}% error")
    st.write(f"This Model scores: {round(test_acc * 100, 2)}% accuracy")

    model.save('brain_tumour_7epochs.h5')

    st.write("Data Analyzing to Graph......")

    logdir = 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    hist = model.fit(xtrain, ytrain,
                     batch_size=16,
                     verbose=1,
                     epochs=2,
                     validation_data=(xtest, ytest),
                     shuffle=False)

    sns.set_style("whitegrid")
    fig1 = sns.lineplot(data=hist.history['loss'], color='teal', label='loss')
    fig1.set_title('Loss', fontsize=20)
    fig1.set_ylabel('Loss')
    fig1.set_xlabel('Epoch')
    fig1.figure.savefig("loss_plot.png")
    st.image("loss_plot.png", use_column_width=True)

    fig2 = sns.lineplot(data=hist.history['accuracy'], color='teal', label='accuracy')
    fig2.set_title('Accuracy', fontsize=20)
    fig2.set_ylabel('Accuracy')
    fig2.set_xlabel('Epoch')
    fig2.figure.savefig("accuracy_plot.png")
    st.image("accuracy_plot.png", use_column_width=True)

    st.write("model saved to folder")

    st.info('Training session Completed!')

########################################################################################################################
if "clicked" not in st.session_state:
    st.session_state.clicked = False


def click_button():
    st.session_state.clicked = True


st.button('Start', on_click=click_button)

if st.session_state.clicked:
    try:
        photo = st.file_uploader("Upload C.T.Scanned Image of Kidney", type=['png', 'jpeg', 'jpg'])
        st.image(photo, caption="Input Image", channels="BGR", width=400)

        ####################################################################################################################
        if st.button('Predict'):
            model = load_model('brain_tumour_7epochs.h5')

            imageinput = Image.open(photo)

            resize = tf.image.resize(imageinput, (256, 256))
            image_input_array = np.array(imageinput)


            pred = model.predict(np.expand_dims(resize / 255, axis=0)).round(2)

            # Spinner/loader code
            with st.spinner("Please wait for result"):
                time.sleep(3)

            st.write("Result is: ")

            # Perform dilation on the image
            kernel = np.ones((5, 5), np.uint8)
            dilated_image = cv2.dilate(image_input_array, kernel, iterations=1)

            # image thresholding
            ret, thresh = cv2.threshold(dilated_image, 200, 255, cv2.THRESH_BINARY_INV)

            # Edge Detection
            edges = cv2.Canny(thresh, 549, 255)

            st.image(thresh, caption="Input Image", channels="BGR", width=400)
            st.image(edges, caption="Input Image", width=400)

            if pred[0] >= 0.5:
                st.write("the input image consists Brain Tumour")
            else:
                st.write("the input image does not contain Brain Tumour")
    except AttributeError as ae:
        st.write("Please Upload the Input Image")
########################################################################################################################
if st.button("Exit"):
    st.success("Thank you")
    time.sleep(6)
    st.markdown("""
        <meta http-equiv="refresh" content="0; url='https://www.google.com'" />
        """, unsafe_allow_html=True)

########################################################################################################################
