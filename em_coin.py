# Implementierung des EM Algorithmus für das 3 Münzen Beispiel

#Quelle : C:\Users\markus.krammer\OneDrive\1_Mathematik\0_Pattern Recognition\EM_Algorithmus



# probability of x for the k component
def pk(x, k):
    resp = 1
    for i in range(len(x)):
        if x[i] == 1:
           resp *= mu[k]
        else:
            resp *= (1 - mu[k])
    if (k == 0):
        return resp * pi
    else:
        return resp * (1 - pi )

def learn():
    #   p(x, y | mu) = p(y | mu) * p(x | y, mu)
    #   p(y | x, mu) = P(x, y | mu) / p(x | mu)

    change = True
    niter = 0


    while change:
        #E Step
        for n in range(N):
            sumz = 0
            for k in range(K):
                z[n][k] = pk(x[n], k)
                #print([n][k])
                sumz += z[n][k]
            for k in range(K):
                z[n][k] /= sumz

        print('\nz:')
        print(z)

        #M Step
        N_m = [0 for k in range(K)]
        z_x = [[0 for d in range(D)] for k in range(K)]

        newpi = [1 / K for k in range(K)]
        newmu = [[1 / D for d in range(D)] for k in range(K)]

        for k in range(K):
            for n in range(N):
                N_m[k] += z[n][k]
                for d in range(D):
                    z_x[k][d] += z[n][k] * x[n][d]
            for d in range(D):
                newmu[k][d] = z_x[k][d] / N_m[k]
            newpi[k] = N_m[k] / N

        print("New_PI:")
        print(newpi)

        print("\nNew_Mu:")
        print(newmu)

        for k in range(K):
            if pi[k] != newpi[k]:
                change = True
                pi[k] = newpi[k]
            for d in range(D):
                if mu[k][d] != newmu[k][d]:
                    change = True
                    mu[k][d] = newmu[k][d]
        niter += 1


    print("Finished in " + str(niter) + " iterations")

# Lauf 1
print("Lauf 1")
# 1 = H, 0 = T
x = [[1,1,1], [0,0,0], [1,1,1], [0,0,0], [1,1,1]]
N = len(x)
D = len(x[0])

pi = 0.3         # Münze 1
mu = [0.3, 0.6]  # Münze 2 oder 3
K = len(mu)
z = [[0 for k in range(K)] for n in range(N)]  # Hilfsmatrix für Zwischenergebnisse

# Sprachliche Auswertung

# Lauf 2
print("Lauf 2")
# 1 =H, 0 = T
x = [[1,1,1], [0,0,0], [1,1,1], [0,0,0]]
N = len(x)
D = len(x[0])

pi = 0.3         # Münze 1
mu = [0.3, 0.6]  # Münze 2 oder 3
K = len(mu)
z = [[0 for k in range(K)] for n in range(N)]  # Hilfsmatrix für Zwischenergebnisse





learn()