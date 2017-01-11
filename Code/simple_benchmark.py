# -*- coding: utf-8 -*-
#!/usr/bin/env python3

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

from sklearn.feature_selection import SelectFromModel

import sys

import random

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
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def import_data(label_type):
    # Write out to csv file
    train = pd.read_csv('../Data/train_'+str(label_type)+'.csv')
    return train

if __name__=="__main__":

    # See if argument given
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
    param['num_class'] = 4
    param['seed'] = 0

    # Set watchlist and number of rounds
    watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
    num_round = 400

    # Progress dictionary
    progress = {}
    # Cross validation - stratified with 10 folds, preserves class proportions
    cvresult = xgb.cv(param, xg_train, num_round, stratified=True, nfold=10, seed = 0,
                      callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
    print(cvresult)
    nrounds = np.linspace(1, num_round+1, num_round)

    # Plot results of simple cross-validation
    plt.errorbar(nrounds, cvresult['train-mlogloss-mean'], yerr=cvresult['train-mlogloss-std'], fmt='o', color='b', label='Training error')
    plt.errorbar(nrounds, cvresult['test-mlogloss-mean'], yerr=cvresult['test-mlogloss-std'], fmt='o', color='r', label='Test error')
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
    plt.show()

    # Make predictions
    yprob = bst.predict( xg_test )

    # Get labels from class probabilities
    ylabel = np.argmax(yprob, axis=1)

    print ('predicting, classification error=%f' % (sum( int(ylabel[i]) != Y_test.as_matrix()[i] for i in range(len(Y_test))) / float(len(Y_test))))
    print ('predicting, classification accuracy= ', 1.0-(sum( int(ylabel[i]) != Y_test.as_matrix()[i] for i in range(len(Y_test))) / float(len(Y_test))))
    misc = np.array([ylabel[i] != Y_test.as_matrix()[i] for i in range(len(Y_test))]).ravel()
    print("Number of misclassified: {0} out of {1}".format(sum( int(ylabel[i]) != Y_test.as_matrix()[i] for i in range(len(Y_test))), len(Y_test)))
    y = np.bincount(Y_test.as_matrix().ravel()[misc])
    ii = np.nonzero(y)[0]
    print(zip(ii, y[ii]))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Y_test, ylabel)
    np.set_printoptions(precision=3)
    class_names = ['RGB', 'RC', 'SC']
    plt.figure(1)
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
    plt.figure(2)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix, with normalization')
    plt.show()


    sel = np.where(np.logical_and(np.max(yprob[misc], axis=1) >= 0.5, np.max(yprob[misc], axis=1) < 0.9))
    #sel = np.where(np.max(yprob[misc], axis=1) > 0.9)

    print(np.max(yprob[misc], axis=1))
    print(len(sel[0]))
    wrong = X_test

    #print("Labels used: ", X_train.columns[selection.get_support()])
    plt.figure(1)
    plt.scatter(X_train['numax'], X_train['denv'], c=Y_train, s=80)
    plt.scatter(X_test['numax'], X_test['denv'], c=Y_test, s=80)
    plt.scatter(X_test['numax'].as_matrix()[misc][sel], X_test['denv'].as_matrix()[misc][sel], c='r', s=80, marker='+')#c=Y_test[misc])#c='yellow')
    #plt.scatter(X_test['numax'].as_matrix()[misc][sel], X_test['denv'][misc].as_matrix()[sel], c='yellow', s=80, marker='+')
    plt.figure(2)
    plt.scatter(X_train['numax'], X_train['Henv'], c=Y_train, s=80)
    plt.scatter(X_test['numax'], X_test['Henv'], c=Y_test, s=80)
    plt.scatter(X_test['numax'].as_matrix()[misc][sel], X_test['Henv'].as_matrix()[misc][sel], c='r', s=80, marker='+')#c=Y_test[misc])#c='yellow')
    #plt.scatter(X_test['numax'].as_matrix()[misc][sel], X_test['denv'][misc].as_matrix()[sel], c='yellow', s=80, marker='+')
    plt.figure(3)
    plt.scatter(X_train['numax'], X_train['Henv'], c=Y_train, s=80)
    plt.scatter(X_test['numax'], X_test['Henv'], c=Y_test, s=80)
    plt.scatter(X_test['numax'].as_matrix()[misc][sel], X_test['Henv'].as_matrix()[misc][sel], c='r', s=80, marker='+')#c=Y_test[misc])#c='yellow')
    #plt.scatter(X_test['numax'].as_matrix()[misc][sel], X_test['denv'][misc].as_matrix()[sel], c='yellow', s=80, marker='+')
    plt.show()

    #miss = X_test.as_matrix()[misc]
    prob = np.max(yprob[misc], axis=1)

    plt.hist(prob, bins=int(np.sqrt(len(prob))), normed=True, histtype='step')
    plt.show()

    # Plot feature importance
    xgb.plot_importance(bst)
    plt.show()
