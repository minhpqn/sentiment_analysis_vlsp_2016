""" Perform quick baseline benchmarck based on bag of words
    for sentiment analysis

    Author: Pham Quang Nhat Minh (FTRI)
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import metrics
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

def get_w(w, tag):
    if tag:
        return w.split('/')[0]
    else:
        return w
    
if __name__ == '__main__':
    os.system('clear')
    tag = False
    
    # use raw data
    # datadir = './data/SA2016-training_data'

    # use data with word segmentation
    datadir = './data/SA2016-training-data-ws'
    
    # filenames = ['SA-training_positive.txt',
    #              'SA-training_negative.txt',
    #              'SA-training_neutral.txt',
    #             ]

    filenames = [
        'train_positive_tokenized.txt',
        'train_negative_tokenized.txt',
        'train_neutral_tokenized.txt',
    ]

    #label_codes = ['pos', 'neg']
    
    label_codes = ['pos', 'neg', 'neutral']

    print("******** Use binary features ********")
    sentences = []
    labels = []
    for i, filename in enumerate(filenames):
        path  = os.path.join(datadir, filename)
        label = label_codes[i]
        f = open(path, 'r')
        for line in f:
            line = line.rstrip()
            if line == '':
                continue
            words = [ get_w(w, tag) for w in line.split()]
            sentences.append( ' '.join( words ) )
            labels.append(label)
            
    y = np.array(labels)
    # count_vect = CountVectorizer( ngram_range = (1,3), binary=True )
    count_vect = CountVectorizer( binary=True )
    X_binary = count_vect.fit_transform( sentences )

    models = [
        LinearSVC(),
        RandomForestClassifier(n_estimators=100, max_depth=None,
                               min_samples_split=1, random_state=0),
    ]
        
    model_names = [
        'Linear SVM',
        'Random Forest',
    ]

    for clf, mdname in zip(models, model_names):
        print('== Use %s method ==' % mdname)
        X = X_binary
        if mdname == 'Gradient Boosting Trees':
            X = X_binary.toarray()
        predicted = cross_validation.cross_val_predict(clf, X, y, cv=10)
        print(metrics.classification_report(y, predicted))
        print

    print
    
    print("******** Use TF-IDF weighting **********")
    
    # count_vect = CountVectorizer(ngram_range = (1,3))
    
    count_vect = CountVectorizer()
    X_count = count_vect.fit_transform( sentences )
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform( X_count )

    for clf, mdname in zip(models, model_names):
        print('== Use %s method ==' % mdname)
        X = X_tfidf
        predicted = cross_validation.cross_val_predict(clf, X, y, cv=10)
        print(metrics.classification_report(y, predicted))
        print



   

    
            
            

            
        

    
        
        
    
    
