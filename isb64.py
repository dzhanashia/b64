import logging
import os.path
import configparser

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

logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

dirname = os.path.dirname(__file__)

config = configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read(os.path.join(dirname, "config.ini"))

if not os.path.exists(os.path.join(dirname, 'trained_clfs')):
    os.makedirs(os.path.join(dirname, 'trained_clfs'))

classifiers = [
        neighbors.KNeighborsClassifier(2, weights='uniform'),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=5, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        SVC(kernel='linear', C=0.025, probability=True)
        ]


def _check_clf(clf):
    """ Trains classifier if there isn't a trained one"""
    if not os.path.exists(os.path.join(dirname, ('trained_clfs/' + clf.__class__.__name__))):
        logging.info('Trained ' + clf.__class__.__name__ + ' not found. Training...')
        _clf_train(clf)


def check_symbols(word):
    """ Checks whether word doesn't have prohibited symbols """
    for symbol in word:
        if not ('A' <= symbol <= 'Z' or 'a' <= symbol <= 'z' or '0' <= symbol <= '9'
                or symbol == '+' or symbol == '=' or symbol == '\\'):
            return False
    return True


def _get_features(word):
    """ Returns list of features"""
    return [
        len(word),
        sum(symbol.isdigit() for symbol in word) / len(word),
        sum(symbol.islower() for symbol in word) / len(word),
        word.count('=')
    ]


def isb64(s, clfs=classifiers):
    """Check if s is b64 encoded

    Arguments:
    s -- string that needs to be checked

    Returns:
    True if s is b64 encoded, False otherwise.
    """
    probs, probs_av = 0, 0

    if check_symbols(s):
        x = np.asarray(_get_features(s)).reshape(1, -1)
        for clf in clfs:
            _check_clf(clf)
            clf = joblib.load(os.path.join(dirname, ('trained_clfs/' + clf.__class__.__name__)))
            probs = np.squeeze(clf.predict_proba(x))[1]
            probs_av += probs
            logging.info(s + ' ' + clf.__class__.__name__ + ': ' + '%.2f' % probs)
    else:
        logging.info(s + ' has prohibited for b64 symbols ' + '%.2f' % probs)

    if probs_av/len(clfs) > 0.5:
        return True
    return False


def pick_out(s, clfs=classifiers):
    """Picks out words in s which are b64 encoded

    Arguments:
    s -- string

    Returns:
    b64 -- list with b64 words
    notb64 -- list with other words
    """
    b64, notb64 = [], []
    for word in s.split():
        if isb64(word, clfs=clfs):
            b64.append(word)
        else:
            notb64.append(word)
    return b64, notb64


def pick_out42(s, clfs=classifiers):
    """Splits s into 42 symbols long fragments and returns indexes of b64 fragments

    Arguments:
    s -- string

    Returns:
    ind -- list with b64 fragments indexes
    """
    ind = []
    index = 0
    for fragment in [s[i:i + 42] for i in range(0, len(s), 42)]:
        if isb64(fragment, clfs=clfs):
            ind.append(index)
        index += 1
    return ind


def delete_clfs(clfs=classifiers):
    """Deletes all existing trained classifiers"""
    for clf in clfs:
        if os.path.exists(os.path.join(dirname, ('trained_clfs/' + clf.__class__.__name__))):
            os.remove(os.path.join(dirname, ('trained_clfs/' + clf.__class__.__name__)))


def _plot_roc(name_clf, y_test, y_pred):
    """Plots ROC Curve and returns ROC Area"""
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

def _get_dataset():
    """Reads dataset from file"""
    y = list(map(int, config.get('dataset', 'y').split()))
    X = list(map(_get_features, config.get('dataset', 'X').split()))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
    return X_train, X_test, y_train, y_test


def _clf_train(clf, X_train=None, y_train=None):
    """Trains and saves classifier"""
    if X_train is None or y_train is None:
        X_train, _, y_train, _ = _get_dataset()
    clf.fit(X_train, y_train)
    joblib.dump(clf, os.path.join(dirname, ('trained_clfs/' + clf.__class__.__name__)))  # Сохранение обученных классификаторов


def _clf_test(clf, X_test, y_test):
    """Test classifier"""
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    roc_auc = _plot_roc(clf.__class__.__name__, y_test, y_pred)
    logging.info(clf.__class__.__name__ + ' Accuracy: ' + '%.2f' % (100 * acc)
                 + ' %. ROC Area: ' + '%5f' % roc_auc + '.')


def main(clfs=classifiers):
    X_train, X_test, y_train, y_test = _get_dataset()
    for clf in clfs:
        _clf_train(clf, X_train, y_train)
        _clf_test(clf, X_test, y_test)


if __name__ == '__main__':
    main()
