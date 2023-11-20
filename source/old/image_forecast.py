import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from glob import glob

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.layers import Dense, Flatten
from keras.callbacks import EarlyStopping


category = ["cats", "dogs"]
EPOCHS = 50
CHANNELS = 1 # grayscale
BATCH_SIZE = 32
STOPPING_PATIENCE = 8
VERBOSE = 1
OPTIMIZER = 'adam'


def read_config(file_name):
    import yaml
    try:
        with open(file_name, "r", encoding="utf-8") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            file = cfg['file']
            TRAINING_DIR = cfg['RAINING_DIR']
            TEST_DIR = cfg['TEST_DIR']
            img_size = cfg['img_size']
            return file, TRAINING_DIR, TEST_DIR, img_size
    except Exception as e:
        print(e)
        return ''


if __name__ == '__main__':
    config_file_name = sys.argv[1]
    print('config_file_name', config_file_name)

    filename, TRAINING_DIR, TEST_DIR,  img_size = read_config(config_file_name)

    df = pd.read_csv(filename)
    df_0 = df[df['category'] == 0]
    df_1 = df[df['category'] == 1]

    generator = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.15,
                                   zoom_range=0.2,
                                   horizontal_flip=True
                                  )

    train_data = generator.flow_from_directory( directory=TRAINING_DIR,
                                                target_size=(img_size, img_size),
                                                color_mode='grayscale',
                                                classes=category,
                                                batch_size=BATCH_SIZE,
                                                )


    test_data = generator.flow_from_directory( directory=TEST_DIR,
                                               target_size=(IMGSIZE, IMGSIZE),
                                               color_mode='grayscale',
                                               classes=category,
                                               batch_size=BATCH_SIZE,
                                               shuffle=False
                                               )

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMGSIZE, IMGSIZE, CHANNELS)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.25))

    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    es = EarlyStopping(patience=STOPPING_PATIENCE,
                       monitor='val_accuracy',
                       mode='max',
                       verbose=1,
                       restore_best_weights=True)

    history = model.fit_generator(train_data,
                                  epochs=EPOCHS,
                                  validation_data=test_data,
                                  shuffle=True,
                                  callbacks=[es]
                                  )

    train_acc = model.evaluate(train_data)
    test_acc = model.evaluate(test_data)