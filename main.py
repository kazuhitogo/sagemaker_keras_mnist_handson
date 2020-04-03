import keras
from tensorflow.keras.layers import Conv2D,PReLU,BatchNormalization,Dropout,MaxPool2D,Flatten,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import categorical_crossentropy
from keras import backend as K
import argparse, os, zipfile
import numpy as np
import tensorflow as tf
from keras import backend as K
from dataloader import ImgFileDataloader
import pandas as pd
from glob import glob

def train(args):
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    epochs = args.epochs
    model_dir = args.model_dir
    train_dir = args.train
    
    # Data Decompression
    decompress(train_dir+"/data.zip" , train_dir+"/")
    
    # Read Data      
    train_img_list = np.array(sorted(glob(os.path.join(train_dir, "data/train_x/*.png"))))
    test_img_list = np.array(sorted(glob(os.path.join(train_dir, "data/test_x/*.png"))))
    train_y = pd.read_csv(os.path.join(train_dir, "data/train_y/train_y.csv"))
    train_y = np.array(train_y["train_y"])
    test_y = pd.read_csv(os.path.join(train_dir, "data/test_y/test_y.csv"))
    test_y = np.array(test_y["test_y"])
    
    train_dataset = ImgFileDataloader(file_list=train_img_list,label_array=train_y,batch_size=train_batch_size,shape=(28,28))
    test_dataset = ImgFileDataloader(file_list=test_img_list,label_array=test_y,batch_size=test_batch_size,shape=(28,28))

    # make model
    mnist_model = Sequential()
    mnist_model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(28,28,1),padding="same"))
    mnist_model.add(PReLU())
    mnist_model.add(Dropout(0.1))
    mnist_model.add(BatchNormalization())
    mnist_model.add(Conv2D(filters=32,kernel_size=(3,3),padding="same"))
    mnist_model.add(PReLU())
    mnist_model.add(Dropout(0.1))
    mnist_model.add(BatchNormalization())
    mnist_model.add(MaxPool2D(pool_size=(2,2),padding="same")) # 28,28 -> 14,14
    mnist_model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same"))
    mnist_model.add(PReLU())
    mnist_model.add(Dropout(0.1))
    mnist_model.add(BatchNormalization())
    mnist_model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same"))
    mnist_model.add(PReLU())
    mnist_model.add(Dropout(0.1))
    mnist_model.add(BatchNormalization())
    mnist_model.add(MaxPool2D(pool_size=(2,2),padding="same")) # 14,14 -> 7,7
    mnist_model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same"))
    mnist_model.add(PReLU())
    mnist_model.add(Dropout(0.1))
    mnist_model.add(BatchNormalization())
    mnist_model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same"))
    mnist_model.add(PReLU())
    mnist_model.add(Dropout(0.1))
    mnist_model.add(BatchNormalization())
    mnist_model.add(MaxPool2D(pool_size=(2,2),padding="same")) # 7,7 -> 4,4
    mnist_model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same"))
    mnist_model.add(PReLU())
    mnist_model.add(Dropout(0.1))
    mnist_model.add(BatchNormalization())
    mnist_model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same"))
    mnist_model.add(PReLU())
    mnist_model.add(Dropout(0.1))
    mnist_model.add(BatchNormalization())
    mnist_model.add(MaxPool2D(pool_size=(2,2),padding="same")) # 4,4 -> 2,2
    mnist_model.add(Conv2D(filters=128,kernel_size=(2,2),padding="same"))
    mnist_model.add(PReLU())
    mnist_model.add(Dropout(0.1))
    mnist_model.add(BatchNormalization())
    mnist_model.add(MaxPool2D(pool_size=(2,2),padding="same")) # 2,2 -> 1,1
    mnist_model.add(Flatten())
    mnist_model.add(Dense(10,activation="softmax"))

    mnist_model.compile(optimizer=Adam(lr=0.0001),metrics=['accuracy'],loss="categorical_crossentropy")

    mnist_model.summary()

    mnist_model.fit_generator(
        generator = train_dataset.data_generator(),
        steps_per_epoch=np.ceil(60000/train_batch_size),
        epochs = epochs,
        verbose=0,
        validation_data = test_dataset.data_generator(),
        validation_steps = np.ceil(60000/test_batch_size),
    )

    model_save(mnist_model,model_dir)

def model_save(model,model_dir):
    sess = K.get_session()
    tf.keras.backend.clear_session()
    tf.saved_model.simple_save(
        sess,
        os.path.join(model_dir, 'model/1'),
        inputs={'inputs': model.input},
        outputs={t.name: t for t in model.outputs})

    
def save(model, model_dir):
    sess = K.get_session()
    tf.saved_model.simple_save(
        sess,
        os.path.join(model_dir, 'model/1'),
        inputs={'inputs': model.input},
        outputs={t.name: t for t in model.outputs})

def decompress(file,dir):
    with zipfile.ZipFile(file) as existing_zip:
        existing_zip.extractall(dir)

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--train-batch-size', type=int, default=128)
    parser.add_argument('--test-batch-size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=1)
    
    # input data and model directories
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    
    args, _ = parser.parse_known_args()
    train(args)