# import the utility libs
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
from skimage.io import imread
from sklearn.model_selection import train_test_split

# import the tensorflow libs
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class HPImageClassifier():
    
    def __init__(self):
        self.train_data_path = "/media/disk3/zeiss/train/"
        self.train_labels_path = "/media/disk3/zeiss/train_labels.csv"
        self.test_data_path = "/media/disk3/zeiss/test/"
        self.test_label_path = "/media/disk3/zeiss/predicted.csv"
        self.model_save_path = "/root/inceptionv3.h5"
        self.model_checkpoint = "/root/checkpoints/"
        
    # load data
    def get_data(self):
        train_df = pd.read_csv(self.train_labels_path)
        train_df['label'] = train_df['label'].astype(str)
        train_df['id'] = train_df['id'].astype(str) + '.tif'
        train, validation = train_test_split(train_df, test_size=0.1)

        return train, validation

    # this method facilitates data augmentation
    def augment_data(self, train, validation):
        train_dgen = ImageDataGenerator(rescale=1./255,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        vertical_flip=True)
        valid_dgen = ImageDataGenerator(rescale=1./255)

        # Extracting the images to a iterator
        train_gen = train_dgen.flow_from_dataframe(
            dataframe=train,
            directory=self.train_data_path,
            x_col='id',
            y_col='label',
            has_ext=False,
            shuffle=True,
            batch_size=64,
            target_size=(96, 96),
            class_mode = 'binary')
        valid_gen = valid_dgen.flow_from_dataframe(
            dataframe=validation,
            directory=self.train_data_path,
            x_col='id',
            y_col='label',
            has_ext=False,
            batch_size=64,
            shuffle=True,
            target_size=(96, 96),
            class_mode = 'binary')

        return train_gen, valid_gen

    # create the model
    def create_model(self):
        # we are using pretrained InceptionV3 model and 'imagenet' weigts
        inception = InceptionV3(weights='imagenet',include_top=False,input_shape=(96,96,3))

        inputs = Input((96,96,3))

        outputs = GlobalAveragePooling2D()(inception(inputs))
        outputs = Dropout(0.5)(outputs)
        outputs = Dense(1, activation="sigmoid")(outputs)

        model = Model(inputs, outputs)

        model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['acc'])

        model.summary()

        return model

    # train the model
    def train_model(self, model, train_gen, valid_gen):
        t_steps = train_gen.n//train_gen.batch_size
        v_steps = valid_gen.n//valid_gen.batch_size

        ck = tf.keras.callbacks.ModelCheckpoint(self.model_checkpoint, 
                                                     save_weights_only=True,
                                                     verbose=1)

        model.fit_generator(train_gen,
                        steps_per_epoch=t_steps ,
                        validation_data=valid_gen,
                        validation_steps=v_steps,
                        epochs=12, callbacks=[ck])

        return model

    def predict(self, model, image):
        img = np.expand_dims(img/255.0, axis=0)
        
        return model.predict(img)[0][0]
    
    def save_model(self, model):
        model.save(self.model_save_path)
        
    def load_model(self, model_path):
        model = tf.keras.models.load_model(model_path)
        return model
        
    # test the model
    def test_model(self, model,batch=5000):
        testing_files = glob(os.path.join(self.test_data_path, '*'))
        submission = pd.DataFrame()
        for index in range(0, len(testing_files), batch):
            data_frame = pd.DataFrame({'path': testing_files[index:index+batch]})
            data_frame['id'] = data_frame.path.map(lambda x: x.split('/')[5].split(".")[0])
            data_frame['image'] = data_frame['path'].map(imread)
            images = np.stack(data_frame.image, axis=0)
            predicted_labels = [model.predict(np.expand_dims(image/255.0, axis=0))[0][0] for image in images]
            predictions = np.array(predicted_labels)
            data_frame['label'] = predictions
            submission = pd.concat([submission, data_frame[["id", "label"]]])
        submission.to_csv(self.test_label_path, index=False, header=True)
