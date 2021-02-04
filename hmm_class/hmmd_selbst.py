import numpy as np
import matplotlib.pyplot as plt

# Erzeugte Markov Matrix d1xd2, alle Zeilen ergeben 1
def random_normalized(d1, d2):
    x= np.random.random((d1,d2))
    return x / x.sum(axis=1,keepdims=True)

class HMM:
    def __init(self,M):
        self.M = M

    def fit(self, X, max_iter=30):
        np.random.seed(123)

        #Grösse des Eingabevokabulars
        V = max(max(x) for x in X) + 1
        N = len(X)

        #Initalisierung
        self.pi = np.ones(self.M) / self.M
        self.A = random_normalized(self.M,self.M)
        self.B = random_normalized(self.M, V)

        costs = []
        for it in range(max_iter):
            if it % 10 == 0:
                print "it:", it

            alphas = []
            betas = []
            P = np.zeros(N)

            for n in range(N):
                x = X[n]
                T = len(x)
                alpha = np.zeros((T,self.M))
                alpha[0] = self.pi * self.B[: x[0]]
                for t in range(1, T)
                    alpha[t] = alpha[t-1].dot(self.A)  * self.B[:, x[t]] # Matrixprodukt
                    P[n] = alpha[-1].sum()
                    alphas.append(alpha)

                beta = np.zeros((T,self.M))
                beta[-1] = 1 # initialer Wert von Beta = 1
                for t in range(T-2,-1,-1) # range start stop step
                    beta[t] = self.A.dot(self.B[:, x[t+1]]) * beta[t+1]  # A * B(Spalte t + 1 ) * beta(Zeile t + 1 )
                    # A(M x M) * B(MxV) * beta(TxM)
                    # A(M x M) * B(M) * beta(M)
                betas.append(beta)

            cost = np.sum(np.Log(P)) # Gesamte Loglikelihood
            costs.append(cost)

            self.pi = np.sum((alphas[n][0] * betas[n][0]) / P[n] for n in range(N) ) / N

            # Re Estimate pi,A,B
            pi = np.sum((alphas[n][0] * betas[n][0])/P[n] for n in range((N)) / N

            den1 = np.zeros(( self.M, 1))
            den2 = np.zeros(( self.M,1 ))

            a_num = 0
            b_num = 0
            for n in range(N)
                x = X[n]
                T = len[x]

                den1 += (alphas[n][n:-1] * betas[n][:-1]).sum(axis = 0,keepdims=TRUE).T / P[n] # loop durch die alphas (ausser letzte Zeile
                den1 += (alphas[n] * betas[n]).sum(axis=0, keepdims=TRUE).T / P[n]  # loop durch die alphas (ausser letzte Zeile

                a_num_n = np.zeros((self.M,self.M))
                for i in range(self.M):
                    for j in range(self.M):
                        for t in range(T-1):
                            a_num-n[i,j] += alphas[n][t,i] * self.A[i,j] * self.B[j,x[t+1]] * betas[n][t+1,j]


















