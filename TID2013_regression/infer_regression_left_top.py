
# Use tensorflow >= 1.10.0 to load this model. I had weird errors.

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
x = Dense(1000, activation='tanh')(x1)
x = Dense(1, activation='linear')(x)

model = Model(inputs=base_model.input, outputs=x)

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
    mean = model.predict(image_batch)

    return mean
'''
# compute predictions for all labels
import pickle
# read image data ------------------------
data_path = '../datasets/distorted_images' # some images do not exist or do not open, I didn't clean dataset
#labels_path = '../datasets/AVA_dataset/AVA.txt'

with open('tid2013_1000.pickle', 'rb') as f:
    data = pickle.load(f)

hists = data['hists']
means = data['means']
stds = data['stds']
filenames = data['filenames']

ids = range(means.shape[0])

label_np = np.empty(shape=(len(ids), hists.shape[1]+1),dtype=float)


# load model train and validation data
label_np[:,1:] = hists
label_np[:,0] = ids

'''

'''
 The TID2008 contains 25 reference images and 3000 distorted images 
(25 reference images x 24 types of distortions x 5 levels of distortions). 
All images are saved in database in Bitmap format without any compression. 
File names are organized in such a manner that they indicate a number of 
the reference image, then a number of distortion's type, and, finally, a 
number of distortion's level: "iXX_YY_Z.bmp". 
'''
'''
n_test = 3*24*5 # 360. take first 3 images as test as in 4PP-EUSR paper


label_train = label_np[n_test:]
label_test = label_np[:n_test]


print(label_train.shape)
print(label_test.shape)



labels_good = []
predictions = []
working_files = []

i = 0
for idx, imgid in enumerate(label_test[:,0]):
    i = i+1
    if i%100==0:
        print(i)
    try:
        filename = path.join(data_path, filenames[int(imgid)])
        mean = test_inference(filename, model=model)
        
    except Exception as e:
        print(e)
        continue
    
    working_files.append([filename, label_test[idx].tolist()])
    labels_good.append(means[idx])
    predictions.append(mean)



labels_good1 = np.array(labels_good)
all_predictions = np.array(predictions)

savemat('tid_wtf_regression.mat', {'labels':labels_good1, 'predictions':all_predictions})
'''


