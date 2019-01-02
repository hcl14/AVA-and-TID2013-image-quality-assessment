# print confusion matrices and built histograms from wtf.mat

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random


import pickle

from scipy.io import loadmat
from scipy.stats import pearsonr, spearmanr

dist_values = np.array([1,2,3,4,5,6,7,8,9,10], dtype=np.float32)
dist_values = np.expand_dims(dist_values, -1)


# required to compare with article
def np_emd_l1(y_true, y_pred):
    cdf_true = np.cumsum(y_true, axis=-1)
    cdf_pred = np.cumsum(y_pred, axis=-1)
    emd = np.mean(np.abs(cdf_true - cdf_pred), axis=-1)
    return np.mean(emd)


def bin_acc(y_true, y_pred):
    num_true = y_true.dot(dist_values)/5.0
    num_pred = y_pred.dot(dist_values)/5.0
    return np.mean(np.equal(np.floor(num_true), np.floor(num_pred)), axis=-1)


data = loadmat('wtf_patches.mat')

label_np = np.squeeze(np.array(data['labels']))
predictions = np.squeeze(data['predictions'])

#with open('test_labels.pickle', 'wb') as handle:
#    pickle.dump(label_np, handle, protocol=pickle.HIGHEST_PROTOCOL)

histograms = label_np[:,1:]
histograms /=  histograms.sum(axis=1)[:,np.newaxis]

# First column is id, other 10 - histogram
label_np[:,1:] = histograms

print('Got labels: {}'.format(len(label_np)))
 
 
rough_total_hist = np.sum(histograms, axis=0)

rough_total_hist /= np.sum(rough_total_hist)


# load predicitons
import pickle

#with open('model_predictions_oversample.pickle', 'rb') as handle:
#    predictions = pickle.load(handle)

# precise mean score distriburtion


mean_scores = histograms.dot(dist_values)
mean_scores_predicted = predictions.dot(dist_values)

def std_dev(x):
    mean = x.dot(dist_values)
    
    # sqrt( E(X^2) - (E(X))^2 )
    return np.sqrt( x.dot(dist_values**2) - mean**2  )
    

std_devs = np.array(list(map(std_dev, histograms)))
std_predicted = np.array(list(map(std_dev, predictions)))



confusion_matrix = np.zeros(shape=(10,10), dtype=int)

acc = []
diffs = []
for idx, value in enumerate(mean_scores):
    confusion_matrix[int(mean_scores_predicted[idx])-1][int(value)-1] += 1
    acc.append(1. - np.abs(mean_scores_predicted[idx] - value)/value)
    diffs.append(np.abs(mean_scores_predicted[idx] - value))
    
    
print(confusion_matrix)
print('accuracy:{}'.format(np.mean(acc)))
print('Binary accuracy:{}'.format(np.mean(bin_acc(label_np[:,1:], predictions))))
print('standard deviation of score differences:{}'.format(np.std(diffs)))
print('LCC: {}'.format(pearsonr(mean_scores_predicted.ravel(), mean_scores.ravel())))
print('SRCC: {}'.format(spearmanr(mean_scores_predicted.ravel(), mean_scores.ravel())))
print('LCC (std dev): {}'.format(pearsonr(std_predicted.ravel(), std_devs.ravel())))
print('SRCC (std dev): {}'.format(spearmanr(std_predicted.ravel(), std_devs.ravel())))
print('EMD L1: {}'.format(np_emd_l1(label_np[:,1:], predictions)))

print('random shuffle:')


# lets random shuffle predictions just for fun and see confusion matrix and accuracy
'''
np.random.shuffle(mean_scores_predicted)
np.random.shuffle(predictions)
confusion_matrix = np.zeros(shape=(10,10), dtype=int)

acc = []
diffs = []
for idx, value in enumerate(mean_scores):
    confusion_matrix[int(mean_scores_predicted[idx])-1][int(value)-1] += 1
    acc.append(1. - np.abs(mean_scores_predicted[idx] - value)/value)
    diffs.append(np.abs(mean_scores_predicted[idx] - value))
    
    
print(confusion_matrix)
print('accuracy:{}'.format(np.mean(acc)))
print('Binary accuracy:{}'.format(np.mean(bin_acc(label_np[:,1:], predictions))))
print('standard deviation of score differences:{}'.format(np.std(diffs)))
print('LCC: {}'.format(pearsonr(mean_scores_predicted.ravel(), mean_scores.ravel())))
print('SRCC: {}'.format(spearmanr(mean_scores_predicted.ravel(), mean_scores.ravel())))
print('EMD L1: {}'.format(np_emd_l1(label_np[:,1:], predictions)))

'''

plt.subplot(1, 2, 1)
out1 = plt.hist(mean_scores, 100, density=0, facecolor='green', alpha=0.5, range=(1,10))
out2 = plt.hist(mean_scores_predicted, 100, density=0, facecolor='red', alpha=0.5, range=(1,10))
plt.xticks(np.arange(1, 10, 1))

plt.xlabel('Scores')
plt.ylabel('No of images')
plt.title('Precise histogram of mean scores')
plt.grid(True)




# precise histogram of standard deviations


plt.subplot(1, 2, 2)
plt.hist(std_devs, 100, density=0, facecolor='green', alpha=0.5, range=(0,2.5))
plt.hist(std_predicted, 100, density=0, facecolor='red', alpha=0.5, range=(0,2.5))
plt.xticks(np.arange(0, 2.5, 0.5))

plt.xlabel('Scores')
plt.ylabel('No of images')
plt.title('Precise histogram of std devs')
plt.grid(True)

plt.show()



