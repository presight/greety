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

import sys
import os
import pickle
import codecs

import pdb

from random import randint

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)

import pandas as pd

import openface
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation

from nolearn.dbn import DBN

from sklearn.metrics import classification_report, accuracy_score

from ConfigParser import SafeConfigParser, NoOptionError


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

    if unknown_reps:
        # Add random sampled unknown embeddings, disabled for now
        unknownImgs = np.load(unknown_reps)

        numUnknown = labels.count(-1)
        numIdentified = len(labels) - numUnknown
        numUnknownAdd = (numIdentified / numIdentities) - numUnknown
        if numUnknownAdd > 0:
            print("Augmenting with {} unknown images".format(numUnknownAdd))
            for i, rep in enumerate(unknownImgs[:numUnknownAdd]):
                embeddings = np.append(embeddings, [rep], axis=0)
                labels.append(-1)
                images = np.append(images, '')
    
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)

    return le, labelsNum, embeddings, images, labels


def print_optimal_min_confidence(success_pred_conf, failed_pred_conf):
    
    success_pred_length = len(success_pred_conf)
    failed_pred_length = len(failed_pred_conf)
    total_pred_length = success_pred_length + failed_pred_length
    

    si = 0  # Success predicate index
    fi = 0  # Failed predicate index
    bmc = 0.0 # Best minimum confidence
    bmcs = 0.0 # Best minimum confidence score

    while True:
        sp = success_pred_conf[si]
        fp = failed_pred_conf[fi]
        
        if sp < fp:
            cp = sp
            si += 1
        elif fp < sp:
            cp = fp
            fi += 1
        else:
            cp = fp
            fi += 1
            si += 1
            
        inaccurate_predictions = si + failed_pred_length - fi
        accurate_predictions = success_pred_length - si + fi

        current_min_prediction_score = float(accurate_predictions) / float(inaccurate_predictions)

        if current_min_prediction_score > bmcs:
            bmc = cp
            bmcs = current_min_prediction_score
            print bmc, bmcs

        if si >= success_pred_length or fi >= failed_pred_length:
            break

    print "Optimal min confidence for run: %s, yielding %s accuracy" % (bmc, bmcs)

    pdb.set_trace()


def evaluate(clf, X, y, embeddings, images, le):
    y_true, y_pred = y, clf.predict(X)

    print "\nClassification report:"
    print classification_report(y_true, y_pred)
    print "Accuracy score: %s" % (accuracy_score(y_true, y_pred))
    
    success_pred_conf = []
    failed_pred_conf = []
    success_pred = []
    failed_pred = []

    for i, image in enumerate(images):
        predictions = clf.predict_proba(X[i].reshape(1, -1)).ravel()
        max_i = np.argmax(predictions)

        if max_i != y[i]:
            e_name = le.inverse_transform(y[i])
            f_name = le.inverse_transform(max_i)
            confidence = predictions[max_i]
            
            failed_pred_conf.append(confidence)
            failed_pred.append("Expected %s, found %s in %s with confidence %s" % (e_name, f_name, image, confidence))
        else:
            name = le.inverse_transform(y[i])
            confidence = predictions[max_i]
            
            success_pred_conf.append(confidence)
            success_pred.append("Found %s in %s with confidence %s" % (name, image, confidence))

    success_below_min_confidence = len([x for x in success_pred_conf if x < min_confidence])
    success_above_min_confidence = len([x for x in success_pred_conf if x > min_confidence])
    failed_below_min_confidence = len([x for x in failed_pred_conf if x < min_confidence])
    failed_above_min_confidence = len([x for x in failed_pred_conf if x > min_confidence])

    print "\nFailed predictions:"
    for p in failed_pred:
        print p

    print "\nSuccessful predictions:"
    for p in success_pred:
        print p

    print "\nSuccessfull predictions below min confidence %s: %s out of %s" % (min_confidence, success_below_min_confidence, success_below_min_confidence+success_above_min_confidence)
    print "\nFailed predictions above min confidence %s: %s out of %s" % (min_confidence, failed_above_min_confidence, failed_below_min_confidence+failed_above_min_confidence)

    success_pred_conf.sort()
    failed_pred_conf.sort()
    
    print "\nSuccessful predictions confidences:"
    print success_pred_conf

    print "\nFailed predictions confidences:"
    print failed_pred_conf

    #print_optimal_min_confidence(success_pred_conf, failed_pred_conf)


def train_clf(dim, X, y, classificator):
    print("Training for {} classes".format(dim[2]))

    if classificator == "DBN":
        clf = DBN(dim,
                  learn_rates=dbn_learn_rates,
                  learn_rate_decays=dbn_learn_rate_decays,
                  epochs=dbn_epochs,
                  minibatch_size=dbn_minibatch_size,
                  verbose=dbn_verbose,
                  dropouts=dbn_dropouts
              )
    elif classificator == "GaussianNB":
        clf = GaussianNB()

    clf.fit(X, y)

    return clf


def save(le, clf):
    fName = classifier_location
    print("\nSaving classifier to '{}'".format(fName))
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)


def train(cross_validate=True, evaluate_result=True):
    le, labelsNum, embeddings, images, labels = get_face_data()

    if cross_validate:
        dim = [embeddings.shape[1], dbn_hidden_dim, len(set(labelsNum))]
        X_train, X_test, y_train, y_test, i_train, i_test = cross_validation.train_test_split(
            embeddings, labelsNum, images, test_size=test_size, random_state=random_state)
        clf = train_clf(dim, X_train, y_train, classificator)
        
        if evaluate_result:
            evaluate(clf, X_test, y_test, embeddings, i_test, le)
    else:
        clf = train_clf(dim, embeddings, labelsNum)

        if evaluate_result:
            evaluate(clf, embeddings, labelsNum, embeddings, images, le)

    save(le, clf)

if __name__ == '__main__':
    conf_file = 'default.conf'

    if len(sys.argv) > 1:
        conf_file = sys.argv[1]

    config = SafeConfigParser()
    config.readfp(codecs.open(conf_file, "r"))

    classifier_location = config.get('Identification', 'classifier')

    labels_data = config.get('Training', 'labels_data')
    reps_data = config.get('Training', 'reps_data')

    try:
        unknown_reps = config.get('Training', 'unknown_reps')
    except NoOptionError:
        unknown_reps = None

    test_size = config.getfloat('Training', 'test_size')
    classificator = config.get('Training', 'classificator')

    dbn_learn_rates = config.getfloat('Training', 'dbn_learn_rates')
    dbn_learn_rate_decays = config.getfloat('Training', 'dbn_learn_rate_decays')
    dbn_epochs = config.getint('Training', 'dbn_epochs')
    dbn_minibatch_size = config.getint('Training', 'dbn_minibatch_size')
    dbn_verbose = config.getint('Training', 'dbn_verbose')
    dbn_dropouts = config.getfloat('Training', 'dbn_dropouts')
    dbn_hidden_dim = config.getint('Training', 'dbn_hidden_dim')

    evaluate_result = config.getboolean('Training', 'evaluate_result')
    cross_validate = config.getboolean('Training', 'cross_validate')

    min_confidence = config.getfloat('Identification', 'min_confidence')

    try:
        random_state = config.getint('Training', 'random_state')
    except NoOptionError:
        random_state = None

    if not random_state:
        random_state = randint(0,10000)

    train(cross_validate, evaluate_result)


    

