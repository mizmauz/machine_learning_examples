# https://deeplearningcourses.com/c/unsupervised-machine-learning-hidden-markov-models-in-python
# https://udemy.com/unsupervised-machine-learning-hidden-markov-models-in-python
# http://lazyprogrammer.me
# Demonstrate how HMMs can be used for classification.
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future



import string
import numpy as np
import matplotlib.pyplot as plt

from hmmd_theano2 import HMM
from sklearn.utils import shuffle
from nltk import pos_tag, word_tokenize

class HMMClassifier:
    def __init__(self):
        pass

    def fit(self, X, Y, V):
        K = len(set(Y)) # number of classes - assume 0..K-1
        N = len(Y)
        self.models = []
        self.priors = []   # Priors für Anwendung der Bayes Rule
        for k in range(K):
            # gather all the training data for this class
            thisX = [x for x, y in zip(X, Y) if y == k]
            C = len(thisX)
            self.priors.append(np.log(C) - np.log(N))

            hmm = HMM(5) # Siehe Lektion 26, wie viele Hidden States?
            # Cross Validation, dann den Wert nehmen, der zum Besten Validation Score führt -> Siehe Scikit Cross Validation
            # Aic: p = Anzahl der Parameter, N=Anzahl der Samples, L = log-Likelihood
            # AIC: Modelle für unterschiedliche Anzahl von M plotten und dann das Modell mit dem höchsten AIC oder BIC verwenden

            hmm.fit(thisX, V=V, print_period=1, learning_rate=1e-2, max_iter=80)

            self.models.append(hmm)

    def score(self, X, Y):
        N = len(Y)
        correct = 0
        for x, y in zip(X, Y):
            lls = [hmm.log_likelihood(x) + prior for hmm, prior in zip(self.models, self.priors)]

            argmax(P(Y=K)|X) = argmax P(X|Y=K)*P(Y=k)
            log P(X|Y=K)+ log(P(Y))

            #hmm.log_likelihood(x) = Posterior
            #prior

            # LLS =
            # This is just how the Bayes classifier works - We want to maximize the posterior:
            #
            # k* = argmax{ P(Y=k | X) } = argmax{ P(X | Y=k)P(Y=k) }
            #
            # It must include both the prior and likelihood terms.



            p = np.argmax(lls)
            if p == y:
                correct += 1
        return float(correct) / N


# def remove_punctuation(s):
#     return s.translate(None, string.punctuation)


# Hier passiert das POS Tagging
# https://www.nltk.org/book/ch05.html
def get_tags(s):
    tuples = pos_tag(word_tokenize(s)) # Tuple: X= wort: Y =tag
    return [y for x, y in tuples]  # gib die y also die Tags zurück

def get_data():
    word2idx = {}
    current_idx = 0
    X = []
    Y = []
    for fn, label in zip(('robert_frost.txt', 'edgar_allan_poe.txt'), (0, 1)):
        count = 0
        for line in open(fn):
            line = line.rstrip()
            if line: #wenn zeile nicht leer
                print(line)
                # tokens = remove_punctuation(line.lower()).split()
                tokens = get_tags(line)
                if len(tokens) > 1:
                    # scan doesn't work nice here, technically could fix...
                    for token in tokens:
                        if token not in word2idx:   # nur wörter, die mehr als 1 mal vorkommen werden berücksichtigt
                            word2idx[token] = current_idx
                            current_idx += 1
                    sequence = np.array([word2idx[w] for w in tokens])
                    X.append(sequence)
                    Y.append(label)
                    count += 1
                    print(count)
                    if count >= 50:
                        break
    print("Vocabulary:", word2idx.keys())
    return X, Y, current_idx
        

def main():

    #Jeweils 50 Zeilen aus den beiden Text = 100 Zeilen;
    # X = Pro Zeile Index der Tags
    # Y = 0,1 pro Zeile

    X, Y, V = get_data()
    print("len(X):", len(X))
    print("Vocabulary size:", V)
    X, Y = shuffle(X, Y)                        # Elemente zufällig vertauschen
    N = 20 # number to test
    Xtrain, Ytrain = X[:-N], Y[:-N]              #XTrain hat Länge 20

    print("XTrain length", len(Xtrain))

    Xtest, Ytest = X[-N:], Y[-N:]                #XTest hat Länge 80

    print("XTest length",len(Xtest))

    model = HMMClassifier()
    model.fit(Xtrain, Ytrain, V)
    print("Score:", model.score(Xtest, Ytest))


if __name__ == '__main__':
    main()
