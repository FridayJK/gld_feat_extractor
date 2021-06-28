# !pip install -q efficientnet
# 1-----------------------
import math, re, os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efn
import random
import pickle
import time
import sys

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


MODEL_PATH = "/workspace/mnt/storage/zhangjunkang/gldv2/model/s2/A100_M3_S2_2/weights.epoch07.loss1.9305.valid_loss0.9265.hdf5"
BATCH = 64
EMB_SIZE = 512
IMAGE_SIZE = [512,512]
EFF_VER = 3
EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3, 
        efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6,efn.EfficientNetB7]


def normalize_image(image):
    image -= tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255])  # RGB
    image /= tf.constant([0.229 * 255, 0.224 * 255, 0.225 * 255])  # RGB
    return image
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = normalize_image(image)
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

def getefn():
    pretrained_model = EFNS[EFF_VER](weights=None, include_top=False ,input_shape=[*IMAGE_SIZE, 3])
    # pretrained_model = EFNS[EFF_VER](weights='./data/GLDv2_models/efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
    pretrained_model.trainable = True
    return pretrained_model
def ArcFaceResNet():
    x= inputs = tf.keras.Input([*IMAGE_SIZE, 3], name='input_image')
    x = getefn()(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dense(EMB_SIZE, activation=None, use_bias=False)(x)
    return tf.keras.Model(inputs, x)

model = ArcFaceResNet()
model.load_weights(MODEL_PATH,by_name=True)
model.summary()

def extract_feat(data_set, imgList = []):
    index_feat = np.empty([len(imgList),EMB_SIZE],dtype="float32")
    input_batch = tf.Variable(np.empty((BATCH, 512,512,3), dtype=np.float32))
    for i, img_path in enumerate(imgList):
        img = open(img_path, 'rb').read()
        input = decode_image(img)
        input_batch[i%BATCH].assign(input)
        if(i==len(imgList)-1):
            output = model(input_batch)
            output = tf.nn.l2_normalize(output, axis=1)
            index_feat[(i-i%BATCH):(i+1),:] = output.numpy()[0:(i%BATCH+1)]
            continue
        if((i+1)%BATCH!=0):
            continue
        output = model(input_batch)
        output = tf.nn.l2_normalize(output, axis=1)
        index_feat[(i-BATCH+1):(i+1),:] = output.numpy()
        if((i+1)%(BATCH*100)==0):
            print("Processed {} rows".format(i))
    return index_feat

if __name__ == '__main__':
    DATA_ROOT_PATH = '/workspace/mnt/storage/zhangjunkang/zjk3/data/GLDv2/'
    DATA_LIST = "./test_list.txt"
   
    images = []
    with open(DATA_LIST) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            images.append(DATA_ROOT_PATH + line.strip())
        
    feat_Path = "./result/feat/"
    os.makedirs(feat_Path,exist_ok=True)

    test_feat = extract_feat("test", images) ###
    np.save(feat_Path+"/testFeat.npy",test_feat)
