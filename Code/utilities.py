# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import division


import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics

from matplotlib.colors import colorConverter
from sklearn.metrics import confusion_matrix

import random

# Set seeds to ensure reproducibility of results
np.random.seed(0)
random.seed(0)

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """



    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    print(cm)
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)

def accuracy_helper(prediction, label):
    print ('Predicting, classification error = %f' % (sum( int(prediction[i]) != label.as_matrix()[i] for i in range(len(label))) / float(len(label))))
    print ('Predicting, classification accuracy = ', 1.0-(sum( int(prediction[i]) != label.as_matrix()[i] for i in range(len(label))) / float(len(label))))
    misc = np.array([prediction[i] != label.as_matrix()[i] for i in range(len(label))]).ravel()
    print("Number of misclassified: {0} out of {1}".format(sum( int(prediction[i]) != label.as_matrix()[i] for i in range(len(label))), len(label)))
    yy = np.bincount(label.as_matrix().ravel()[misc])
    ii = np.nonzero(yy)[0]
    print("Breakdown of misclassified labels:\n")
    print(zip(ii, yy[ii]))
    return misc

def compute_precision_recall(cnf_matrix):

    precision = [cnf_matrix[i,i] / np.sum(cnf_matrix[:,i]) for i in range(len(cnf_matrix))]
    recall = [cnf_matrix[i,i] / np.sum(cnf_matrix[i,:]) for i in range(len(cnf_matrix))]
    print("Precision: ", precision)
    print("Recall: ", recall)
    return precision, recall

def precision_recall(prediction, label):

    n_classes = len(np.unique(label))
    precision = {}
    recall = {}
    average_precision = {}
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(label[:,i])
