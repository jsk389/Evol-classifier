# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import division

import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn import mixture
import os
from operator import itemgetter

from matplotlib.colors import colorConverter
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, confusion_matrix
import sklearn.metrics
from scipy.stats import randint as sp_randint
import itertools
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
import pickle
from sklearn.feature_selection import SelectFromModel

import sys

import random
from utilities import *

np.random.seed(0)

random.seed(0)

def import_data(label_type):
    # Write out to csv file
    train = pd.read_csv('../Data/train_'+str(label_type)+'.csv')
    return train

def import_Kep_K2(label_type):
    # Write out to csv file
    train = pd.read_csv('../Data/train_Kep_as_K2_'+str(label_type)+'.csv')
    return train

def create_realisation(x):
    x = x.as_matrix()
    n_params = np.shape(x)[1] // 2
    n_stars = len(x)
    new_x = np.zeros([n_stars, n_params])
    for i in range(n_params):
        new_x[:,i] = x[:,(i*2)] + np.random.randn(n_stars) * x[:,(i*2)+1]
    df = pd.DataFrame(new_x, columns=['hsig1', 'b1', 'c1', 'hsig2', 'b2', 'c2', 'numax', 'denv', 'Henv', 'alpha', 'beta', 'c3'])
    return df

if __name__=="__main__":

    # See if argument given
    try:
        label_type = str(sys.argv[1])
    except:
        label_type = None
    print(label_type)

    # Load in model
    bst = pickle.load(open("xgboost_simple.dat", "rb"))


    XK2 = import_Kep_K2(label_type)
    y = XK2[['evol_overall']]
    X = XK2.drop(['evol_overall'], axis=1)

    # Make predictions
    n_reals = 1000
    predictions = np.zeros([len(X), n_reals, 3])
    for i in range(n_reals):
        print("Realisation {0} of {1}".format(i+1, n_reals))
        realisation = create_realisation(X)

        xg_K2 = xgb.DMatrix(realisation, label=y)

        yprob = bst.predict( xg_K2 )

        predictions[:,i,:] = yprob

    # Compute predictions from distribution
    probs = np.median(predictions, axis=1)

    # Compute labels
    ylabel = np.argmax(np.median(predictions, axis=1), axis=1)

    # Compute classification accuracy
    misc = accuracy_helper(ylabel, y)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y, ylabel)

    compute_precision_recall(cnf_matrix)

    np.set_printoptions(precision=3)
    class_names = ['RGB', 'RC', 'SC']
    plt.figure(1)
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
    plt.figure(2)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix, with normalization')
    plt.show()



    plt.figure(1)
    plt.scatter(X['numax'], X['denv'], c=y, s=80)
    plt.scatter(X['numax'].as_matrix()[misc], X['denv'].as_matrix()[misc], c='yellow', s=80, marker='+')#c=Y_test[misc])#c='yellow')
    plt.xlabel(r'$\nu_{\mathrm{max}}$ ($\mu$Hz)', fontsize=18)
    plt.ylabel(r'$\delta\nu_{\mathrm{env}}$ ($\mu$Hz)', fontsize=18)
    plt.figure(2)
    plt.scatter(X['numax'], X['Henv'], c=y, s=80)
    plt.scatter(X['numax'].as_matrix()[misc], X['Henv'].as_matrix()[misc], c='r', s=80, marker='+')#c=Y_test[misc])#c='yellow')
    #plt.scatter(X_test['numax'].as_matrix()[misc][sel], X_test['denv'][misc].as_matrix()[sel], c='yellow', s=80, marker='+')
    plt.figure(3)
    plt.scatter(X['numax'], X['hsig1'], c=y, s=80)
    plt.scatter(X['numax'].as_matrix()[misc] X['hsig1'].as_matrix()[misc], c='r', s=80, marker='+')#c=Y_test[misc])#c='yellow')
    #plt.scatter(X_test['numax'].as_matrix()[misc][sel], X_test['denv'][misc].as_matrix()[sel], c='yellow', s=80, marker='+')
    plt.show()
