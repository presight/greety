#!/usr/bin/env python2
#
# Based on openface/demos/web/websocket-server.py and openface/demos/classifier.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import os
import pickle

import pdb

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import openface

from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn import metrics

from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from nolearn.dbn import DBN

from sklearn.metrics import classification_report, accuracy_score



if __name__ == '__main__':
    print("Loading embeddings.")
    fname = "generated/labels.csv"
    labels_data = pd.read_csv(fname, header=None).as_matrix()
    labels = labels_data[:, 1]
    images = labels_data[:, 1]
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory.
    fname = "generated/reps.csv"
    embeddings = pd.read_csv(fname, header=None).as_matrix()

    numIdentities = len(set(labels + [-1])) - 1

    # Add random sampled unknown embeddings
    unknownImgs = np.load("./openface/demos/web/unknown.npy")

    numUnknown = labels.count(-1)
    numIdentified = len(labels) - numUnknown
    numUnknownAdd = (numIdentified / numIdentities) - numUnknown
    if numUnknownAdd > 0:
        print("+ Augmenting with {} unknown images.".format(numUnknownAdd))
        for i, rep in enumerate(unknownImgs[:numUnknownAdd]):
            # print(rep)
            embeddings = np.append(embeddings, [rep], axis=0)
            labels.append(-1)
    

    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)
    print("Training for {} classes.".format(nClasses))

    #param_grid = [
    #    {'C': [1, 10, 100, 1000],
    #     'kernel': ['linear']},
    #    {'C': [1, 10, 100, 1000],
    #     'kernel': ['linear']},
    #    {'C': [1, 10, 100, 1000],
    #     'gamma': [0.001, 0.0001],
    #     'kernel': ['rbf']}
    #]

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        embeddings, labelsNum, test_size=0.4, random_state=0)


    #le_y_train = LabelEncoder().fit(labels)
    #labelsNum_y_train = le.transform(labels)

    #le_y_test = LabelEncoder().fit(y_test)
    #labelsNum_y_test = le.transform(y_test)

    
    #svm = GridSearchCV(SVC(C=1), param_grid, cv=5,verbose=50).fit(embeddings, labels)
    
    #svm = GridSearchCV(SVC(C=1), param_grid,verbose=50)#.fit(embeddings, labels)


    #svm = GridSearchCV(SVC(C=1), param_grid, cv=5,verbose=50).fit(X_train, y_train)
    #print svm.score(X_test, y_test)


    clfChoice = 'DBN'
    ladim = 0


    if clfChoice == 'LinearSvm':
        clf = SVC(C=1, kernel='linear', probability=True)
    elif clfChoice == 'GMM':  # Doesn't work best
        clf = GMM(n_components=nClasses)

        # ref:
        # http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#example-classification-plot-classifier-comparison-py
    elif clfChoice == 'RadialSvm':  # Radial Basis Function kernel
        # works better with C = 1 and gamma = 2
        clf = SVC(C=1, kernel='rbf', probability=True, gamma=2)
    elif clfChoice == 'DecisionTree':  # Doesn't work best
        clf = DecisionTreeClassifier(max_depth=20)
    elif clfChoice == 'GaussianNB':
        clf = GaussianNB()
        
        # ref: https://jessesw.com/Deep-Learning/
    elif clfChoice == 'DBN':
        verbose = 1
        #pdb.set_trace(
        clf = DBN([embeddings.shape[1], 500, len(set(labelsNum))],  # i/p nodes, hidden nodes, o/p nodes
                      learn_rates=0.3,
                      # Smaller steps mean a possibly more accurate result, but the
                      # training will take longer
                      learn_rate_decays=0.9,
                      # a factor the initial learning rate will be multiplied by
                      # after each iteration of the training
                      epochs=300,  # no of iternation
                      # dropouts = 0.25, # Express the percentage of nodes that
                      # will be randomly dropped as a decimal.
                      verbose=verbose)

    if ladim > 0:
        clf_final = clf
        clf = Pipeline([('lda', LDA(n_components=args.ldaDim)),
                        ('clf', clf_final)])

    #clf.fit(embeddings, labelsNum)
    clf.fit(X_train, y_train)

    if clfChoice == "DBN":
        #y_true, y_pred = y_test, clf.predict(X_test)
        ##scores = cross_validation.cross_val_score(clf, embeddings, labels, cv=5)
        #predicted = cross_validation.cross_val_predict(clf, embeddings, labels, cv=10)

        #print classification_report(y_true, y_pred)
        #print accuracy_score(y_true, y_pred)
        #pdb.set_trace()
    

        for i, image in enumerate(images):
            #pdb.set_trace()
            predictions = clf.predict_proba(embeddings[i].reshape(1, -1)).ravel()
            max_i = np.argmax(predictions)
        
            name = le.inverse_transform(max_i)
            confidence = predictions[max_i]

            if name != labels[i]:
                #pdb.set_trace()
                print "expected %s, found %s in %s with score %s" % (labels[i], name, image, confidence)
            #else:
                #print "correctly predicted %s in %s with %s probability" % (name, image, confidence)

        #for i, image in enumerate(images):
        #    if labels[i] != predicted[i]:
        #        print "expected %s, found %s in %s with score %s" % (labels[i], predicted[i], image, scores[i])

    else:
        scores = cross_validation.cross_val_score(clf, embeddings, labels, cv=5)
        predicted = cross_validation.cross_val_predict(clf, embeddings, labels, cv=10)
        metrics = metrics.accuracy_score(labels, predicted) 

        print predicted
        print metrics

    #pdb.set_trace()
            
    #print scores
    #print labels


    

    fName = "generated/classifier.pkl"
    print("Saving classifier to '{}'".format(fName))
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)
