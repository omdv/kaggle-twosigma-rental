"""
http://www.kaggle.com/c/bioresponse/forums/t/1889/\
question-about-the-process-of-ensemble-learning/10950#post10950
https://github.com/emanuele/kaggle_pbr/blob/master/blend.py
"""
from __future__ import division
import numpy as np
import load_data
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':

    np.random.seed(2017)  # seed to shuffle the train set

    n_folds = 10
    verbose = True
    shuffle = False
    ifSparse = False

    X, y, X_submission = load_data.load(ifSparse)
    y = y.values.reshape(1,-1)[0]
    X = X.values
    X_submission = X_submission.values

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(StratifiedKFold(y, n_folds))

    clfs = [RandomForestClassifier(n_estimators=400, n_jobs=-1,\
                criterion='gini'),
            RandomForestClassifier(n_estimators=400, n_jobs=-1,\
                criterion='entropy'),
            ExtraTreesClassifier(n_estimators=400, n_jobs=-1,\
                criterion='gini'),
            ExtraTreesClassifier(n_estimators=400, n_jobs=-1,\
                criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5,\
                max_depth=6, n_estimators=50)]

    print "Creating train and test sets for blending."

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:, 1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:, 1]
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

    print
    print "Blending."
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:, 1]

    # print "Saving Results."
    # tmp = np.vstack([range(1, len(y_submission)+1), y_submission]).T
    # np.savetxt(fname='submission.csv', X=tmp, fmt='%d,%0.9f',
    #            header='MoleculeId,PredictedProbability', comments='')