#Datenset Target: https://debuggercafe.com/image-classification-with-mnist-dataset/
#http://blog.manfredas.com/expectation-maximization-tutorial/
#https://mlbhanuyerra.github.io/2018-01-28-Handwritten-Digits_Mixture-of-Bernoulli-Distributions/
#https://codereview.stackexchange.com/questions/132252/pattern-recognition-and-machine-learning-bernoulli-mixture-model


from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

import sklearn.datasets
from sklearn.datasets import fetch_openml

def show_pictures(X):
    digit = X[0]
    digit_pixels = digit.reshape(28, 28)
    plt.subplot(131)
    plt.imshow(digit_pixels)
    plt.axis('off')
    digit = X[1]
    digit_pixels = digit.reshape(28, 28)
    plt.subplot(132)
    plt.imshow(digit_pixels)
    plt.axis('off')
    digit = X[2]
    digit_pixels = digit.reshape(28, 28)
    plt.subplot(133)
    plt.imshow(digit_pixels)
    plt.axis('off')
    plt.show()

# Training 60.0000, Test 10.0000
def divide_test_training(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]

def show(image):
    '''
    Function to plot the MNIST data
    '''
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = ax.imshow(image, cmap=plt.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    plt.show()


def bernoulli(data, means):
    '''To compute the probability of x for each bernouli distribution
    data = N X D matrix
    means = K X D matrix
    prob (result) = N X K matrix
    '''
    N = len(data)
    K = len(means)
    # compute prob(x/mean)
    # prob[i, k] for ith data point, and kth cluster/mixture distribution
    prob = np.zeros((N, K))

    for i in range(N):
        for k in range(K):
            prob[i, k] = np.prod((means[k] ** data[i]) * ((1 - means[k]) ** (1 - data[i])))

    return prob


def respBernoulli(data, weights, means):
    '''To compute responsibilities, or posterior probability p(z/x)
    data = N X D matrix
    weights = K dimensional vector
    means = K X D matrix
    prob or resp (result) = N X K matrix
    '''
    # step 1
    # calculate the p(x/means)
    prob = bernoulli(data, means)

    # step 2
    # calculate the numerator of the resp.s
    prob = prob * weights

    # step 3
    # calcualte the denominator of the resp.s
    row_sums = prob.sum(axis=1)[:, np.newaxis]

    # step 4
    # calculate the resp.s
    try:
        prob = prob / row_sums
        return prob
    except ZeroDivisionError:
        print("Division by zero occured in reponsibility calculations!")


def bernoulliMStep(data, resp):
    '''Re-estimate the parameters using the current responsibilities
    data = N X D matrix
    resp = N X K matrix
    return revised weights (K vector) and means (K X D matrix)
    '''
    N = len(data)
    D = len(data[0])
    K = len(resp[0])

    Nk = np.sum(resp, axis=0)
    mus = np.empty((K, D))

    for k in range(K):
        mus[k] = np.sum(resp[:, k][:, np.newaxis] * data, axis=0)  # sum is over N data points
        try:
            mus[k] = mus[k] / Nk[k]
        except ZeroDivisionError:
            print("Division by zero occured in Mixture of Bernoulli Dist M-Step!")
            break

    return (Nk / N, mus)


def llBernoulli(data, weights, means):
    '''To compute expectation of the loglikelihood of Mixture of Beroullie distributions
    Since computing E(LL) requires computing responsibilities, this function does a double-duty
    to return responsibilities too
    '''
    N = len(data)
    K = len(means)

    resp = respBernoulli(data, weights, means)

    ll = 0
    for i in range(N):
        sumK = 0
        for k in range(K):
            try:
                temp1 = ((means[k] ** data[i]) * ((1 - means[k]) ** (1 - data[i])))
                temp1 = np.log(temp1.clip(min=1e-50))

            except:
                print("Problem computing log(probability)")
            sumK += resp[i, k] * (np.log(weights[k]) + np.sum(temp1))
        ll += sumK

    return (ll, resp)


def mixOfBernoulliEM(data, init_weights, init_means, maxiters=1000, relgap=1e-4, verbose=False):
    '''EM algo fo Mixture of Bernoulli Distributions'''
    N = len(data)
    D = len(data[0])
    K = len(init_means)

    # initalize
    weights = init_weights[:]
    means = init_means[:]
    ll, resp = llBernoulli(data, weights, means)
    ll_old = ll

    for i in range(maxiters):
        if verbose and (i % 5 == 0):
            print("iteration {}:".format(i))
            print("   {}:".format(weights))
            print("   {:.6}".format(ll))

        # E Step: calculate resps
        # Skip, rolled into log likelihood calc
        # For 0th step, done as part of initialization

        # M Step
        weights, means = bernoulliMStep(data, resp)

        # convergence check
        ll, resp = llBernoulli(data, weights, means)
        if np.abs(ll - ll_old) < relgap:
            print("Relative gap:{:.8} at iternations {}".format(ll - ll_old, i))
            break
        else:
            ll_old = ll

    return (weights, means, resp)

#shuffle funktioniert nicht, deshalb diese Funktion
def pickData_fix(mnist,digits, N):
    array = np.zeros(shape=(N,784), dtype='int')
    count = 0
    res = list(map(str, digits))  # this call works for Python 2.x as well as for 3.x

    for i in range(len(mnist.target)):
        if mnist.target[i] in res:
            curdata = mnist.data[i]
            array[count] = curdata
            count = count + 1
            if count == N:
                return array

def pickData(mnist, digits, N):
    sData, sTarget = shuffle(mnist, mnist.target, random_state=30)
    returnData = np.array([sData[i] for i in range(len(sData)) if sTarget[i] in digits])
    return shuffle(returnData, n_samples=N, random_state=30)

def experiments(expData,digits, K, N, iters=50):
    '''
    Picks N random points of the selected 'digits' from MNIST data set and
    fits a model using Mixture of Bernoulli distributions.
    And returns the weights and means.
    '''

    D = len(expData[0])

    initWts = np.random.uniform(.25, .75, K) #K Beispiele aus der Verteilung ziehen
    tot = np.sum(initWts)
    initWts = initWts / tot

    # initMeans = np.random.rand(10,D)
    initMeans = np.full((K, D), 1.0 / K)   # nxk Matrix gefüllt mit 1/K

    return mixOfBernoulliEM(expData, initWts, initMeans, maxiters=iters, relgap=1e-15)

mnist = fetch_openml('mnist_784', version=1)
print(mnist.keys())
print(mnist.data.shape)

'''
MNIST data is in grey scale [0, 255].
Convert it to a binary scale using a threshold of 128.
'''
mnist.data = (mnist.data / 128).astype('int')

basicdata = [2,3,9]
expData = pickData_fix(mnist, basicdata, 1000)

finWeights, finMeans, resp = experiments(expData,[2,3,9], 3, 1000)

#Zeige einige Zahlen (resp NxK), 1000x3
from random import seed
from random import randint

# seed random number generator
seed(1)
zufallbild = randint(0, 999)
show(expData[zufallbild].reshape(28,28))

wert = np.amax(resp[zufallbild]) #axis 0 = col 1 = row
result = np.where(resp[zufallbild] == np.amax(resp[zufallbild]))
y = result[0][0]

print("Die Zahl heißt",resp[zufallbild][y],"Soll sein",basicdata[y])
print(finMeans[result])


#[show(finMeans[i].reshape(28,28)) for i in range(len(finMeans))]

#show_pictures(mnist.data)
