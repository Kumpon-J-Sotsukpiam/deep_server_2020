# from keras.applications.vgg16 import VGG16
# from keras_applications.vgg16 import preprocess_input
from app.keras_vggface.vggface import VGGFace
from app.keras_vggface.utils import preprocess_input
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Lambda, Dropout, Activation
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator, image
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adadelta
import matplotlib.pyplot as plt
from glob import glob

from app.config import IMAGE_SIZE, BATCH_SIZE, EPOCH
import os
import cv2
import numpy as np
import shutil

class VGGModel(object):
    def __init__(self):
        self.model = None
        self.labels = []

    def build_model(self, nb_classes):
        vggModel = VGGFace(model='vgg16', weights='vggface',
                           input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        for layer in vggModel.layers:
            layer.trainable = False
        last_layer = vggModel.get_layer('pool5').output
        # full connection layers
        x = Flatten(name='flatten')(last_layer)
        x = Dense(1024, activation='relu', name='fc6')(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu', name='fc7')(x)
        x = Dropout(0.5)(x)
        out = Dense(nb_classes, activation='softmax', name='fc8')(x)
        # full connection layers
        self.model = Model(vggModel.input, out)
        print('Build Model !')
        return True

    def train(self, dataTemp_path, batch_size=BATCH_SIZE, nb_epoch=EPOCH):
        subpath = ['data/', 'eval/']
        train_path = os.path.join(dataTemp_path, subpath[0])
        eval_path = os.path.join(dataTemp_path, subpath[1])
        datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
        train_generator = datagen.flow_from_directory(
                                train_path,
                                target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                batch_size=BATCH_SIZE,
                                class_mode='sparse',
                                color_mode='rgb')
        valid_generator = datagen.flow_from_directory(
                                directory=eval_path,
                                target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                color_mode='rgb',
                                batch_size=BATCH_SIZE,
                                class_mode='sparse',
                                shuffle=True)
        labels = (train_generator.class_indices)
        #self.labels = dict((v, k) for k, v in labels.items())
        for k in labels.items():
            self.labels.append(k[0])
        print(self.labels)
        #simple early stopping
        es = EarlyStopping(monitor='val_acc', mode='max', verbose=1)
        self.model.compile(loss='sparse_categorical_crossentropy',
                         optimizer=SGD(lr=1e-4, momentum=0.9),
                         metrics=['accuracy'])
        self.model.fit_generator(
                        train_generator,
                        validation_data=valid_generator,
                        steps_per_epoch=train_generator.n//BATCH_SIZE,
                        validation_steps=valid_generator.n//BATCH_SIZE,
                        epochs=nb_epoch,
                        #callbacks=[es]
                        )
        print('success training model !')
        return True

    def save(self, file_path="vgg_face.h5"):
        self.model.save(file_path)

    def load(self, file_path="vgg_face.h5"):
        try:
            self.model = load_model(file_path)
        except IOError as e:
            return False, "File not found"
        return True, "Loaded Model"

    def predict(self, image):
        # preProcess = image
        # preProcess = cv2.resize(preProcess,(IMAGE_SIZE,IMAGE_SIZE))
        return self.model.predict(image)
