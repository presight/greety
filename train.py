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
    numUnknownAdd = (numIdentified / numIdentities) - numUnknown
    if numUnknownAdd > 0:
        print("Augmenting with {} unknown images".format(numUnknownAdd))
        for i, rep in enumerate(unknownImgs[:numUnknownAdd]):
            # print(rep)
            embeddings = np.append(embeddings, [rep], axis=0)
            labels.append(-1)
    
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)

    return le, labelsNum, embeddings, images, labels

def evaluate(dbn, X, y, le, embeddings, images, labels):
    y_true, y_pred = y, dbn.predict(X)

    print "\nClassification report:"
    print classification_report(y_true, y_pred)
    print "Accuracy score: %s" % (accuracy_score(y_true, y_pred))

    # Printing failed predictions, this is good to get a sense of what images the net has problems recognizing
    print "\nFailed predictions:"
    for i, image in enumerate(images):
        predictions = dbn.predict_proba(embeddings[i].reshape(1, -1)).ravel()
        max_i = np.argmax(predictions)
        
        name = le.inverse_transform(max_i)
        confidence = predictions[max_i]
        
        if name != labels[i]:
            print "expected %s, found %s in %s with confidence %s" % (labels[i], name, image, confidence)
    

def train_dbn(X, y, labelsNum):
    no_labels = len(set(labelsNum))
    print("Training for {} classes".format(no_labels))

    dbn = DBN([X.shape[1], 1500, no_labels],
              learn_rates=0.3,
              learn_rate_decays=0.9,
              epochs=100,
              minibatch_size=6,
              dropouts=0.15)

    dbn.fit(X, y)

    return dbn

def save(le, dbn):
    fName = "generated/classifier.pkl"
    print("\nSaving classifier to '{}'".format(fName))
    with open(fName, 'w') as f:
        pickle.dump((le, dbn), f)


def train(cross_validate=False, evaluate_result=True):
    le, labelsNum, embeddings, images, labels = get_face_data()
    
    if cross_validate:
        train_X, test_X, train_y, test_y = cross_validation.train_test_split(
            embeddings, labelsNum, test_size=0.4, random_state=0)
        dbn = train_dbn(train_X, train_y, labelsNum)
        
        if evaluate_result:
            evaluate(dbn, test_X, test_y, le, embeddings, images, labels)
    else:
        dbn = train_dbn(embeddings, labelsNum, labelsNum)

        if evaluate_result:
            evaluate(dbn, embeddings, labelsNum, le, embeddings, images, labels)

    save(le, dbn)

if __name__ == '__main__':
    train(evaluate_result=True, cross_validate=True)


    

