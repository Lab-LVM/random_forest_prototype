import numpy as np
import math


def getEntropy(labels, base=2):

    """ Computes entropy of label distribution. """
    n_labels = len(labels)
    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.0
    # Compute entropy
    base = 2 if base is None else base
    for i in probs:
        ent -= i * math.log(i, base)

    return ent


def getBinaryPseudoLabels(labels, uniform=True):
    ul = np.unique(labels)
    np.random.shuffle(ul)
    ul_1 = ul[0:round(len(ul)/2)]
    return [1 if i in ul_1 else 0 for i in labels]
