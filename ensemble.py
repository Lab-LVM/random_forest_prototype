import sys
import util
import numpy as np
import random
from joblib import Parallel, delayed
from sklearn import svm

class DecisionTree:
    def __init__(self, maxDepth=8, nt=50, split_fn='F', repeat=10, minDataNum=2, dim_SVM = -1):
        assert maxDepth > 0
        assert maxDepth < 21
        assert nt > 0
        assert (split_fn == 'F') or (split_fn == 'S' and dim_SVM > 0)
        assert repeat > 0
        assert minDataNum > 1

        self.maxDepth = maxDepth
        self.nt = nt
        self.split_fn = split_fn
        self.repeat = repeat
        self.minDataNum = minDataNum
        self.dim_SVM = dim_SVM
        self.node = {}

    def build_node(self, features, labels, index, depth, maxLabel):
        if depth >= self.maxDepth or features.shape[0] < self.minDataNum or len(np.unique(labels)) <= 1:
            self.node.update({str(index) + '_node': 'leaf'})

            PMF = np.divide(np.histogram(labels, range(0, maxLabel + 2)), labels.size)
            self.node.update({str(index) + '_PMF': PMF[0]})
            return

        base_ent = util.getEntropy(labels)
        maxIG = -10e8
        left_index = []
        right_index = []
        maxDim = []
        maxMV = []
        maxCLF = []

        if self.split_fn == 'F':
            dims = np.random.permutation(features.shape[1])

        for i in range(0, self.repeat):
            if self.split_fn == 'F':
                dim = dims[i]
                mv = np.median(features[:, dim])
                l_idx = mv < features[:, dim]
                r_idx = ~l_idx

            elif self.split_fn == 'S':
                dim = np.random.permutation(features.shape[1])
                dim = dim[0:self.dim_SVM]
                f = features[:, dim]
                pseudo_labels = util.getBinaryPseudoLabels(labels)
                clf = svm.LinearSVC()
                clf.fit(f,pseudo_labels)
                score = clf.decision_function(f)
                mv = np.median(score)

                l_idx = mv<score
                r_idx = ~l_idx

            l_lbs = labels[l_idx]
            r_lbs = labels[r_idx]

            IG = base_ent - (util.getEntropy(l_lbs) * l_lbs.size / labels.size \
                             + util.getEntropy(r_lbs) * r_lbs.size / labels.size)

            if IG > maxIG:
                maxMV = mv
                maxDim = dim
                left_index = l_idx
                right_index = r_idx
                maxIG = IG
                if self.split_fn == 'S':
                    maxCLF= clf

        if np.sum(left_index) == 0 or np.sum(right_index) == 0:
            self.node.update({str(index) + '_node': 'leaf'})
            PMF = np.divide(np.histogram(labels, range(0, maxLabel + 2)), labels.size)
            self.node.update({str(index) + '_PMF': PMF[0]})
            return

        self.node.update({str(index) + '_node': 'normal'})
        self.node.update({str(index) + '_mv': maxMV})
        self.node.update({str(index) + '_dim': maxDim})
        if self.split_fn == 'S':
            self.node.update({str(index) + '_CLF': maxCLF})

        left_labels = labels[left_index]
        right_labels = labels[right_index]

        left_features = features[left_index, :]
        right_features = features[right_index, :]

        self.build_node(left_features, left_labels, index * 2 + 1, depth + 1, maxLabel)
        self.build_node(right_features, right_labels, index * 2 + 2, depth + 1, maxLabel)

    def build_tree(self, features, labels):
        depth = 0
        labels = np.ravel(labels)
        prior = np.divide(np.histogram(labels, range(0, int(np.max(labels)) + 2)), labels.size)
        self.prior = prior[0]
        self.build_node(features, labels, 0, depth, int(np.max(labels)))

    def predict(self, feature, index=0):
        assert index >= 0
        node = self.node[str(index)+'_node']

        if node=="normal":
            mv = self.node[str(index) + '_mv']
            dim = self.node[str(index) + '_dim']

            if self.split_fn == 'F':
                child = mv > feature[dim]
            elif self.split_fn == 'S':
                clf = self.node[str(index) + '_CLF']
                f = feature[dim]
                f = np.reshape(f, (1, -1))
                score = clf.decision_function(f)
                child = mv > score

            return self.predict(feature, index * 2 + int(child) + 1)

        elif node=="leaf":
            PMF = self.node[str(index) + '_PMF']
            PMF = np.divide(PMF, np.add(self.prior, 10e-16))
            return PMF


class RandomForest:
    def __init__(self, maxDepth=8, nt=50, split_fn='F', repeat=10, minDataNum=2, dim_SVM = -1):
        assert maxDepth > 0
        assert maxDepth < 21
        assert nt > 0
        assert (split_fn == 'F') or (split_fn == 'S' and dim_SVM > 0)
        assert repeat > 0
        assert minDataNum > 1

        self.maxDepth = maxDepth
        self.nt = nt
        self.split_fn = split_fn
        self.repeat = repeat
        self.minDataNum = minDataNum
        self.trees = [None] * nt
        self.dim_SVM = dim_SVM

    def build_forest(self, features, labels):
        labels = np.ravel(labels)
        assert features.shape[0] == labels.size

        #self.trees = [
        #    build_tree_thread(self.maxDepth, self.nt, self.split_fn, self.repeat, self.minDataNum, self.dim_SVM,
        #                      features, labels) for i in range(0, self.nt)]

        self.trees = Parallel(n_jobs=4)( delayed(build_tree_thread)\
            (self.maxDepth, self.nt, self.split_fn, self.repeat, self.minDataNum, self.dim_SVM, features, labels)\
            for i in range(0, self.nt))



    def predict(self, feature, index=0):
        PMF = 0
        for i in range(0, self.nt):
            PMF = PMF + self.trees[i].predict(feature, index=0)

        return PMF


def build_tree_thread( maxDepth, nt, split_fn, repeat, minDataNum, dim_SVM, features, labels):
    tree = DecisionTree(maxDepth, nt, split_fn, repeat, minDataNum, dim_SVM)
    tree.build_tree(features, labels)
    return tree
    #self.trees[i] = tree#.append(tree)
