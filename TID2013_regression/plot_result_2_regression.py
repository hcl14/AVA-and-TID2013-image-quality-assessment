# print confusion matrices and built histograms from wtf.mat

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random


import pickle

from scipy.io import loadmat
from scipy.stats import pearsonr, spearmanr

# required to compare with article
def np_emd_l1(y_true, y_pred):
    cdf_true = np.cumsum(y_true, axis=-1)
    cdf_pred = np.cumsum(y_pred, axis=-1)
    emd = np.mean(np.abs(cdf_true - cdf_pred), axis=-1)
    return np.mean(emd)


data = loadmat('tid_wtf_regression.mat')

label_np = np.squeeze(np.array(data['labels']))
predictions = np.squeeze(data['predictions'])

mean_scores = label_np
mean_scores_predicted = predictions

confusion_matrix = np.zeros(shape=(10,10), dtype=int)

acc = []
diffs = []
for idx, value in enumerate(mean_scores):
    confusion_matrix[int(mean_scores_predicted[idx])-1][int(value)-1] += 1
    acc.append(1. - np.abs(mean_scores_predicted[idx] - value)/value)
    diffs.append(np.abs(mean_scores_predicted[idx] - value))
    
    
print(confusion_matrix)
print('accuracy:{}'.format(np.mean(acc)))
print('standard deviation of score differences:{}'.format(np.std(diffs)))
print('LCC: {}'.format(pearsonr(mean_scores_predicted.ravel(), mean_scores.ravel())))
print('SRCC: {}'.format(spearmanr(mean_scores_predicted.ravel(), mean_scores.ravel())))
#print('EMD L1: {}'.format(np_emd_l1(label_np, predictions)))



out1 = plt.hist(mean_scores, 100, density=0, facecolor='green', alpha=0.5, range=(1,10))
out2 = plt.hist(mean_scores_predicted, 100, density=0, facecolor='red', alpha=0.5, range=(1,10))
plt.xticks(np.arange(1, 10, 1))

plt.xlabel('Scores')
plt.ylabel('No of images')
plt.title('Precise histogram of mean scores')
plt.grid(True)




plt.show()



