# test jpeg image distortions from folder
import os


from os import path
import numpy as np
import pandas as pd

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Concatenate
import tensorflow as tf
from scipy.io import savemat

try: 
    base_model = keras.applications.mobilenetv2.MobileNetV2((224,224,3), pooling='avg', weights=None, include_top=True)
except:
    base_model = keras.applications.mobilenet_v2.MobileNetV2((224,224,3), pooling='avg', weights=None, include_top=True)




x1 = Dropout(rate=0.75)(base_model.layers[-2].output)
x2 = Dense(1000, activation='tanh')(x1)
x = Dense(1, activation='linear')(x2)

model = Model(inputs=base_model.input, outputs=[x, x2])

print(model.summary())

# __file__ work when running not with 'python3 -i', just python
model.load_weights(path.join(path.dirname(path.abspath(__file__)),'weights-tid2013-regression-194-0.95.hdf5'))


# Normalize according to ImageNet
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
# for each channel do input[channel] = (input[channel] - mean[channel]) / std[channel
def normalize_img(img):    
    img /= 255.    
    for channel in range(3):
        mean = IMAGE_NET_MEAN[channel]
        std = IMAGE_NET_STD[channel]
        img[:,:, channel] = (img[:,:, channel] - mean)/std
        
    return img

    
def crop_center(img, height=224, width=224):
    
    h = img.shape[0]-height
    w = img.shape[1]-width
    y0 = h//2
    x0 = w//2
    
    return img[y0:y0+height , x0:x0+width, :]


# predict scores  - use this function to evaluate your images
def test_inference(filename, model=model):
    # load an image in PIL format
    original = load_img(filename)
    numpy_image = img_to_array(original)
    
    numpy_image_n = normalize_img(numpy_image.copy())
    
    # predict on central patch
    numpy_image_n = crop_center(numpy_image_n)

    # (1, 224, 224, 3)
    image_batch = np.expand_dims(numpy_image_n, axis=0)
    
    # get the predicted probabilities for each class
    mean, features = model.predict(image_batch)

    return mean, features






filenames = [f for f in os.listdir('test_distortions')]

for filename in filenames:
    mean1, _ = test_inference('test_distortions/'+filename)
    print('file:{}, score:{}'.format(filename, mean1))



