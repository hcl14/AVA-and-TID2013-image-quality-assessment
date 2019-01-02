
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




#x1 = Dropout(rate=0.75)(base_model.layers[-2].output)
x = Dense(10, activation='softmax')(base_model.layers[-2].output)#(x1)

model = Model(inputs=base_model.input, outputs=x)

#print(model.summary())

# __file__ work when running not with 'python3 -i', just python
model.load_weights(path.join(path.dirname(path.abspath(__file__)),'weights-patches-continue6-12-0.77.hdf5'))


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

# store elements of distribution to compute means, etc
dist_values = np.array([1,2,3,4,5,6,7,8,9,10], dtype=np.float32)
dist_values = np.expand_dims(dist_values, -1)

def mean_std(x):
    mean = x.dot(dist_values)
    # sqrt( E(X^2) - (E(X))^2 )
    return mean, np.sqrt(x.dot(dist_values**2) - mean**2 )


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
    predictions = model.predict(image_batch)

    mean, std = mean_std(predictions)
    
    return mean, std, predictions

# compute predictions for all labels - uncomment if you want to generate predictions for entire test set
'''
# read image data ------------------------
import json

with open('ava_labels_test.json', 'r') as f:
    data = json.load(f)
    
label_test = []
for value in data:
    label_test.append([value["image_id"]]+value["label"])
    
label_test = np.array(label_test, dtype=float)

histograms = label_test[:,1:]
histograms /=  histograms.sum(axis=1)[:,np.newaxis]

# First column is id, other 10 - histogram
label_test[:,1:] = histograms

data_path = '../datasets/AVA_dataset/images/images' # some images do not exist or do not open, I didn't clean dataset

labels_good = []
predictions = []
working_files = []

i = 0
for idx, imgid in enumerate(label_test[:,0]):
    i = i+1
    if i%100==0:
        print(i)
    try:
        filename = path.join(data_path, str(int(imgid))+'.jpg')
        mean, std, pred = test_inference(filename, model=model)
        
    except Exception as e:
        print(e)
        continue
    
    working_files.append([filename, label_test[idx].tolist()])
    labels_good.append(label_test[idx])
    predictions.append(pred)



labels_good1 = np.array(labels_good)
all_predictions = np.array(predictions)

with open('ava_labels_test_working_files.json', 'w') as f:
    json.dump(working_files,f)

savemat('wtf_patches.mat', {'labels':labels_good1, 'predictions':all_predictions})
    
'''

