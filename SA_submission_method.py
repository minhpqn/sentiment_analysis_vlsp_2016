"""
  The script used to generate official submission for SA VLSP 2016 Challenge
"""

from __read_data import read_train_data, read_test_data
import os
import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn import metrics

def train(train_sens, y_train):
    C_OPTIONS = np.logspace(-9,3,15)
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,3))),
                         ('tfidf', TfidfTransformer()),
                         ('clf', LinearSVC()),])
    parameters = {
        'clf__C': C_OPTIONS
    }

    score = 'f1_macro'
    
    print("\nTuning parameters for Linear SVM\n")
    gs_clf = GridSearchCV(text_clf, parameters, cv = 5, scoring=score,
                          n_jobs=-1)
    gs_clf.fit(train_sens, y_train)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
    print("Best score (Grid search): %f" % gs_clf.best_score_)

    print("\nTuning parameters for Random Forest\n")
    rfc = RandomForestClassifier(max_depth=None,
                                  min_samples_split=1, random_state=0)
    rfc_pip = Pipeline([('vect', CountVectorizer(ngram_range=(1,3))),
                         ('tfidf', TfidfTransformer()),
                         ('clf', rfc),])
    rfc_parameters = {
        'clf__n_estimators': [50, 100, 150, 200, 250, 300, 400, 500, 700]
    }

    gs_rfc = GridSearchCV(rfc_pip, rfc_parameters, cv = 5, scoring=score,
                          n_jobs=-1)
    gs_rfc.fit(train_sens, y_train)
    for param_name in sorted(rfc_parameters.keys()):
        print("%s: %r" % (param_name, gs_rfc.best_params_[param_name]))
    print("Best score (Grid search): %f" % gs_rfc.best_score_)

    print("\nEnsemble learning\n")        

    clf1 = RandomForestClassifier(n_estimators=gs_rfc.best_params_['clf__n_estimators'], max_depth=None, min_samples_split=1, random_state=0)
    clf2 = LinearSVC(C=gs_clf.best_params_['clf__C'])
    clf3 = MultinomialNB()
    ensemble_clf = VotingClassifier(
            estimators=[('rd', clf1), ('linearSVM', clf2),
                        ('mnb', clf3),
                       ],
            voting='hard')

    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,3))),
                         ('tfidf', TfidfTransformer()),
                         ('clf', ensemble_clf),])

    predicted = cross_validation.cross_val_predict(text_clf, train_sens, y_train, cv=5)
    print(metrics.classification_report(y_train, predicted))
    
    text_clf.fit(train_sens, y_train)
    return text_clf

def predict(text_clf, test_sens):
    return text_clf.predict(test_sens)

def main(args):
    output='./minhpqn_ftri_SA_submission.txt'
    if len(args) > 0:
        output = args[0]
    train_sens, y_train = read_train_data()
    test_sens = read_test_data()

    text_clf = train(train_sens, y_train)
    predicted_labels = predict(text_clf, test_sens)
    f = open(output, 'w')
    for sen, label in zip(test_sens, predicted_labels):
        f.write('%s\n' % sen)
        f.write('%s\n' % label)
    f.close()
    
if __name__ == '__main__':
    os.system('clear')
    args=sys.argv[1:]
    main(args)
    
