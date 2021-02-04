import numpy as np

# Arrays manipulieren
# Matrix Gleichung lösen
# Lineare Regression

A = np.array([[1,-1,2],[3,2,0]]) # 2,3 Matrix

A[-1] = 5 # Letzte Zeile

v = np.array([[2],[1],[3]]) # Vector

v = np.transpose(np.array([[2,1,3]]))

print(A[1,2]) # 0 ausgeben

col = A[0:1,1:2] # 2. Spalte und 1. Zeile rausnehmen

print(col)



# Gleichung Ax = b lösen
A = np.array([[2,1,-2],[3,0,1],[1,1,-1]])
b = np.transpose(np.array([[-3,5,-2]]))
x = np.linalg.solve(A,b)
print("Solution for Ax=b ", x)

# Matrixmultiplikation mit dot
X = np.array([0,1,2])
T = len(X)

for t in range(0, T):
    C = A[:,X[t]] # Spalte
    print("Das ist C ", C)


print("A ", A[1]) # 2. Zeile aus A

# Beispiel Lineare Regression (Windsor Haus Preise )
#
# import csv
# import numpy as np
#
# def readData():
#     X = []
#     y = []
#     with open('Housing.csv') as f:
#         rdr = csv.reader(f)
#         # Skip the header row
#         next(rdr)
#         # Read X and y
#         for line in rdr:
#             xline = [1.0]
#             for s in line[:-1]:
#                 xline.append(float(s))
#             X.append(xline)
#             y.append(float(line[-1]))
#     return (X,y)
#
# X0,y0 = readData()
# # Convert all but the last 10 rows of the raw data to numpy arrays
# d = len(X0)-10
# X = np.array(X0[:d])
# y = np.transpose(np.array([y0[:d]]))
#
# # Compute beta
# Xt = np.transpose(X)
# XtX = np.dot(Xt,X)
# Xty = np.dot(Xt,y)
# beta = np.linalg.solve(XtX,Xty)
# print(beta)
#
# # Make predictions for the last 10 rows in the data set
# for data,actual in zip(X0[d:],y0[d:]):
#     x = np.array([data])
#     prediction = np.dot(x,beta)
#     print('prediction = '+str(prediction[0,0])+' actual = '+str(actual))