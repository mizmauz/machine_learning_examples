import numpy as np
from sklearn.utils import shuffle

def testarray():
    # generate random integer values
    from random import seed
    from random import randint
    # seed random number generator
    seed(1)
    # generate some integers
    for _ in range(20):
        value = randint(0, 10)
        print(value)

    #Fix Multidimensional
    array1 = [1, 2, 3]
    array2 = [4, 5, 6]

    array3 = [array1, array2]
    print(array3[1][2])

    #Multi Dimensional Array
    b = [[randint(0, 255) for y in range(3)]
               for x in range(10)]
    print("Multidimensiona\n")
    print(b)

    # create multi-dim array by providing shape
    matrix = np.zeros(shape=(2,5),dtype='int')
    print("Multidimensiona1\n")
    print(matrix)

def testindex():
    from random import seed
    from random import randint
    b = [[randint(0, 255) for y in range(3)]
         for x in range(10)]

    seed(1)
    x = randint(0, 10)

    maxwert = np.amax(b[x])  # axis 0 = col 1 = row
    yint = np.where(b[x] == np.amax(b[x]))
    y = yint[0][0]

    print('Wert',maxwert,'Result',y,'Wert',b[x][y])

def pickData(mnist, digits, N):
    sData, sTarget = shuffle(mnist, mnist.target, random_state=30)
    returnData = np.array([sData[i] for i in range(len(sData)) if sTarget[i] in digits])
    return shuffle(returnData, n_samples=N, random_state=30)

#shuffle funktioniert nicht, deshalb diese Funktion
def pickData_fix(mnist, digits, N):
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

def from_file():
    from sklearn.datasets import fetch_openml
    data_path = "C:/Users/markus.krammer/OneDrive/1_Mathematik/0_Pattern Recognition/0_Scripts/Python_Scripts/dataset"
    mnist = fetch_openml('mnist_784', data_home=data_path)
    pickdata = pickData_fix(mnist,[2,3,9],20)
    print(pickdata)

def other():
    array = np.full((0,np.full()),0.0)
    arrayadd = [[0,0,0]]
    for i in range(10):
        array = array + arrayadd
        print(array)


#testarray()
#from_file()
testindex()
