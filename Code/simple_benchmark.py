# -*- coding: utf-8 -*-
#!/usr/bin/env python3
##
# This code is designed to provide a simple benchmark for th classification
# task
##

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
import pickle
from sklearn.preprocessing import PolynomialFeatures

from sklearn.feature_selection import SelectFromModel

import sys

import random
from utilities import *

# Set seeds to ensure reproducibility of results
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

if __name__=="__main__":

    # See if argument given to decide whether use consensus labels or those
    # from Elsworth (2017)
    try:
        label_type = str(sys.argv[1])
    except:
        label_type = None
    print(label_type)
    train = import_data(label_type)

    # Set label as target
    y = train[['evol_overall']]
    # Drop target from feature array
    X = train.drop(['evol_overall'], axis=1)

    # Train, test split
    X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.33)

    # Create Dmatrix for use with xgboost
    xg_train = xgb.DMatrix(X_train, label=Y_train)
    xg_test =  xgb.DMatrix(X_test, label=Y_test)
    xg_full = xgb.DMatrix(X, label=y)

    # setup parameters for xgboost
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softprob'
    param['eval_metric'] = 'mlogloss'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['max_depth'] = 3
    param['silent'] = 1
    param['nthread'] = 4
    param['num_class'] = len(np.unique(y))
    param['seed'] = 0

    # Set watchlist and number of rounds
    watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
    num_round = 400

    # Progress dictionary
    progress = {}
    # Cross validation - stratified with 10 folds, preserves class proportions
    cvresult = xgb.cv(param, xg_train, num_round, stratified=True, nfold=10, seed = 0,
                      callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

    nrounds = np.linspace(1, num_round+1, num_round)

    # Plot results of simple cross-validation
    plt.errorbar(nrounds, cvresult['train-mlogloss-mean'], yerr=cvresult['train-mlogloss-std'], fmt='o', color='b', label='Training error')
    plt.errorbar(nrounds, cvresult['test-mlogloss-mean'], yerr=cvresult['test-mlogloss-std'], fmt='o', color='r', label='Test error')
    plt.xlabel(r'Epoch', fontsize=18)
    plt.ylabel(r'Log loss', fontsize=18)
    plt.show()


    # do the same thing again, but output probabilities
    param['objective'] = 'multi:softprob'
    param['seed'] = 0
    # Number of rounds used that minimises test error in cross-validation used for last training of model
    bst = xgb.train(param, xg_train, int(cvresult[cvresult['test-mlogloss-mean'] == np.min(cvresult['test-mlogloss-mean'])].index[0]), watchlist, early_stopping_rounds=50, evals_result=progress)
    print(progress)

    # Print test and train errors
    plt.plot(progress['train']['mlogloss'], color='b')
    plt.plot(progress['test']['mlogloss'], color='b')
    plt.xlabel(r'Epoch', fontsize=18)
    plt.ylabel(r'Log loss', fontsize=18)
    plt.show()

    # Make predictions
    yprob = bst.predict( xg_test )

    # Get labels from class probabilities
    ylabel = np.argmax(yprob, axis=1)

    # Compute classification accuracy
    misc = accuracy_helper(ylabel, Y_test)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Y_test, ylabel)

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


    # Plot Henv and denv marking misclassified stars
    plt.figure(1)
    plt.scatter(X_train['numax'], X_train['denv'], c=Y_train, s=80)
    plt.scatter(X_test['numax'], X_test['denv'], c=Y_test, s=80)
    plt.scatter(X_test['numax'].as_matrix()[misc], X_test['denv'].as_matrix()[misc], c='yellow', s=80, marker='+')#c=Y_test[misc])#c='yellow')
    plt.xlabel(r'$\nu_{\mathrm{max}}$ ($\mu$Hz)', fontsize=18)
    plt.ylabel(r'$\delta\nu_{\mathrm{env}}$ ($\mu$Hz)', fontsize=18)
    plt.figure(2)
    plt.scatter(X_train['numax'], X_train['Henv'], c=Y_train, s=80)
    plt.scatter(X_test['numax'], X_test['Henv'], c=Y_test, s=80)
    plt.scatter(X_test['numax'].as_matrix()[misc], X_test['Henv'].as_matrix()[misc], c='yellow', s=80, marker='+')#c=Y_test[misc])#c='yellow')

    plt.show()


    # Plot feature importance
    xgb.plot_importance(bst)
    plt.show()

    # Write out model
    pickle.dump(bst, open("xgboost_simple.dat", "wb"))
