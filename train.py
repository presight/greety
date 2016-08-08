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

from random import randint

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)

import pandas as pd

import openface
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation

from nolearn.dbn import DBN

from sklearn.metrics import classification_report, accuracy_score

def get_face_data():
    fname = "generated/labels.csv"
    labels_data = pd.read_csv(fname, header=None).as_matrix()
    images = labels_data[:, 1]
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, images)))  # Get the directory.
    fname = "generated/reps.csv"
    embeddings = pd.read_csv(fname, header=None).as_matrix()

    numIdentities = len(set(labels + [-1])) - 1

    # Add random sampled unknown embeddings
    unknownImgs = np.load("./openface/demos/web/unknown.npy")

    numUnknown = labels.count(-1)
    numIdentified = len(labels) - numUnknown
    numUnknownAdd = 0 #(numIdentified / numIdentities) - numUnknown
    if numUnknownAdd > 0:
        print("Augmenting with {} unknown images".format(numUnknownAdd))
        for i, rep in enumerate(unknownImgs[:numUnknownAdd]):
            # print(rep)
            embeddings = np.append(embeddings, [rep], axis=0)
            labels.append(-1)
    
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)

    return le, labelsNum, embeddings, images, labels

def evaluate(clf, X, y, embeddings, images, le):
    y_true, y_pred = y, clf.predict(X)

    print "\nClassification report:"
    print classification_report(y_true, y_pred)
    print "Accuracy score: %s" % (accuracy_score(y_true, y_pred))
    print y_true
    print y_pred

    # Printing failed predictions, this is good to get a sense of what images the net has problems recognizing
    print "\nFailed predictions:"

    success_pred_conf = []
    failed_pred_conf = []

    for i, image in enumerate(images):
        predictions = clf.predict_proba(X[i].reshape(1, -1)).ravel()
        max_i = np.argmax(predictions)

        if max_i != y[i]:
            e_name = le.inverse_transform(y[i])
            f_name = le.inverse_transform(max_i)
            confidence = predictions[max_i]
            
            failed_pred_conf.append(confidence)
            print "expected %s, found %s in %s with confidence %s" % (e_name, f_name, image, confidence)
        else:
            name = le.inverse_transform(y[i])
            confidence = predictions[max_i]
            
            success_pred_conf.append(confidence)
            #print "found %s in %s with confidence %s" % (name, image, confidence)

    #success_pred_conf.sort()
    #failed_pred_conf.sort()
    #print success_pred_conf
    #print failed_pred_conf


def train_clf(dim, X, y, type="DBN"):
    print("Training for {} classes".format(dim[2]))

    if type == "DBN":
        clf = DBN(dim,
                  learn_rates=0.005,
                  learn_rate_decays=1,
                  epochs=500,
                  minibatch_size=6,
                  verbose=0,
                  dropouts=0.15
              )
    elif type == "GaussianNB":
        clf = GaussianNB()

    clf.fit(X, y)

    return clf


def save(le, clf):
    fName = "generated/classifier.pkl"
    print("\nSaving classifier to '{}'".format(fName))
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)


def train(cross_validate=True, evaluate_result=True):
    le, labelsNum, embeddings, images, labels = get_face_data()
    dim = [embeddings.shape[1], 24, len(set(labelsNum))]

    if cross_validate:
        X_train, X_test, y_train, y_test, i_train, i_test = cross_validation.train_test_split(
            embeddings, labelsNum, images, test_size=0.4, random_state=4)#randint(0,1000))
        clf = train_clf(dim, X_train, y_train)
        
        if evaluate_result:
            evaluate(clf, X_test, y_test, embeddings, i_test, le)
    else:
        clf = train_clf(dim, embeddings, labelsNum)

        if evaluate_result:
            evaluate(clf, embeddings, labelsNum, embeddings, images, le)

    save(le, clf)

if __name__ == '__main__':
    train(evaluate_result=True, cross_validate=True)


    

