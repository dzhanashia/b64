import logging
import os.path

import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn import neighbors
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

classifiers = [
        SVC(kernel='linear', C=0.025, probability=True),
        neighbors.KNeighborsClassifier(2, weights='uniform'),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=5, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()
        ]

logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _name(clf):
    return str(clf).partition('(')[0]


def _initialize(clf):
    if not os.path.isfile(_name(clf)):
        logging.info('Trained ' + _name(clf) + ' not found. Training...')
        main(test=False, clfs=(clf,))


def symbols_allowed(s):
    s = s[:32]
    for symb in s:
        if not ('A' <= symb <= 'Z' or 'a' <= symb <= 'z' or '0' <= symb <= '9' or symb == '+' or symb == '=' or symb == '\h'[:-1]):
            return False
    return True


def _get_features(s):
    s = s[:32]
    return [len(s), sum(c.isdigit() for c in s) / len(s), sum(c.islower() for c in s) / len(s), s.count('=')]


def isb64(s, log=True, clfs=classifiers):
    probs, probs_av = 0, 0

    if not isinstance(s, str):
        s = s.read()

    if symbols_allowed(s):
        x = np.asarray(_get_features(s)).reshape(1, -1)
        for clf in clfs:
            _initialize(clf)
            clf = joblib.load(_name(clf))
            probs = np.squeeze(clf.predict_proba(x))[1]
            probs_av += probs
            if log:
                logging.info(s + ' ' + _name(clf) + ': ' + '%.2f' % probs)
    else:
        if log:
            logging.info(s + ' has prohibited for b64 symbols ' + '%.2f' % probs)

    if probs_av/len(clfs) > 0.5:
        return True
    return False


def pick_out(s, index=True, clfs=classifiers):
    b64, notb64 = [], []
    ind = []

    if not isinstance(s, str):
        s = s.read()

    s = s.split()
    i = 0
    for word in s:
        i += 1
        if isb64(word, log=False, clfs=clfs):
            b64.append(word)
            ind.append(i)
        else:
            notb64.append(word)
    if index:
        return ind
    else:
        return b64, notb64


def delete_clfs(clfs=classifiers):
    for clf in clfs:
        if os.path.isfile(_name(clf)):
            os.remove(_name(clf))


def _create_xy(fill):
    X, y = [], []
    for row in fill:
        y.append(int(row[0]))
        X.append(_get_features(row[2:-1]))
    return X, y


def _plot_roc(name_clf, y_test, y_pred):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.6f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='-.')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC ' + name_clf)
    plt.legend(loc='lower right')
    plt.show()

    return roc_auc


def _clf_train(clfs, X_train, y_train):
    for clf in clfs:
        clf.fit(X_train, y_train)
        joblib.dump(clf, _name(clf))  # Сохранение обученных классификаторов


def _clf_test(clfs, X_test, y_test):
    for clf in clfs:
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        roc_auc = _plot_roc(_name(clf), y_test, y_pred)
        logging.info(_name(clf) + ' Accuracy: ' + '%.2f' % (100 * acc) + ' %. ROC Area: ' + '%5f' % roc_auc + '.')


def main(test=True, clfs=classifiers):
    fill = open('dataset.txt', 'r', encoding='utf-8')
    X, y = _create_xy(fill)
    fill.close()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
    _clf_train(clfs, X_train, y_train)
    if test:
        _clf_test(clfs, X_test, y_test)


if __name__ == '__main__':
    main()
