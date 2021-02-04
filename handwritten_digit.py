# Author: Markus Krammer
# Bernoulli Mixture Modell mit Testdaten aus dem Minst Datenset

#Quellen
#http://fourier.eng.hmc.edu/e176/lectures/ch9/node19.html
#http://blog.manfredas.com/expectation-maximization-tutorial/
#https://mlbhanuyerra.github.io/2018-01-28-Handwritten-Digits_Mixture-of-Bernoulli-Distributions/
#https://codereview.stackexchange.com/questions/132252/pattern-recognition-and-machine-learning-bernoulli-mixture-model


#Vorlage: HandwrittenDigits_Bernoulli.py

#Laden des Datensets
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



def completelikelihood(data,means,weights):
    mainprob = 0

    N = len(data)
    K = len(means)

    prob = np.zeros((N, K))

    D = len(data)
    for i in range(N):
        for k in range(K):
            #mit Submethode
            prob[i,k] = ( weights[k] * np.log( bernoulli(data, means, weights, i,k) ))
            #mit np.prod
            #mainprob = np.prod((means[k] ** data[i]) * ((1 - means[k]) ** (1 - data[i])))
            mainprob += prob[i,k]

    print("P(X,Y|mü,init)\n")
    print(mainprob)
    print("Probmatrix\n")
    print(prob)


def bernoulli(data,means,weights,xi,k):
    '''To compute the probability of x for each bernouli distribution
    data = N X D matrix
    means = K X D matrix
    prob (result) = N X K matrix
    '''

    # compute prob(x/mean)
    # prob[i, k] for ith data point, and kth cluster/mixture distribution
    #prob = np.zeros((N, K))

    prob = 1
    for d in range(len(data[xi])):
        prob = prob * ( means[k][d]**data[xi][d]) * (1-means[k][d])**(1-data[xi][d])

    return prob


#Neuberechnung der latenten Werte Z|X,mü
#def e_step(data,weights,means):
    # N = LEN(DATA)
    # K = LEN(MEANS)
    # D = LEN(DATA[0])
    #
    # FOR N IN RANGE(N):
    #     FOR K IN RANGE(K):
    #         FOR I IN RANGE(D):
    #             Z[N][K] = ( MEANS[K][I]**DATA[N][I] )*( 1 - MEANS[K][I]**DATA[N][I])
    #             /

#Take an image an Predict the Cluster
#https://codereview.stackexchange.com/questions/132252/pattern-recognition-and-machine-learning-bernoulli-mixture-model
def prediction_single(images,mu,k):
# Input: The input images NxD
# mu: Parameters for Bernoulli Distribution for each pixel (KxD)
# phi: Mixing Koeffizeients (kx1)
# k : number of mixtures
# Ouput: Cluster Nx1 Wahrscheinlichstes Cluster für das Bild

    N = np.size(images, 1);
    ClusterSum = np.zeros(N, K)
    Temp1 = mu;
    Temp2 = 1 - Temp1;
    for n = 1: N
        for k = 1: K
        # In:  http://blog.manfredas.com/expectation-maximization-tutorial/
        # They; used; the; sum; of; mu; 's, but to be honest I don'; t; know; why

        ClusterSum(n, k) = prod(Temp1(k, images(n,:) == 1)) *prod(Temp2(k, images(n,:) == 0));
        end
    end
[~, Cluster] = max(ClusterSum, [], 2);



def initialize():
    #expData = mnist.data[0]
    expData = [[1.0,1.0,0.0,0.0,]]

    D = len(expData[0])
    N = len(expData)
    K = 2

    initWts = np.random.uniform(.25, .75, K)  # Initialgewichtung
    tot = np.sum(initWts)
    initWts = initWts / tot

    # initMeans = np.random.rand(10,D)
    initMeans = np.full((K, D), 1.0 / K)  # nxk Matrix gefüllt mit 1/K

    print("InitMeans\n")
    print(initMeans)
    print("InitWeights\n")
    print(initWts)


    completelikelihood(expData,initMeans,initWts)


#mnist = fetch_openml('mnist_784', version=1)

#Dimensionen des Datensets
#print(mnist.keys())
#print(mnist.data.shape)

#Testbilder Anzeigen
#show_pictures(mnist.data)

#K = 2  # Anzahl der Cluster
#D = 4  #mnist.data.shape[1]  # Dimension der Bilder
#N = 1  # Anzahl der Bilder
initialize( )


#Aufbau der Formel
#https://mlbhanuyerra.github.io/2018-01-28-Handwritten-Digits_Mixture-of-Bernoulli-Distributions/


