import matplotlib.pyplot as plt
from pylab import *
from random import *



def readArchive(path, xColumn=1, yColumn=2):
    x = []
    y = []

    with open(path) as file:
        for line in file:
            aux = line.split()
            if (len(aux) != 0):
                x.append(float(aux[xColumn]))
                y.append(float(aux[yColumn]))

    return x, y


def readtable(name):
    f = open(name, 'r')
    lines = f.readlines()
    result = []
    for x in lines:
        result.append(x)
    f.close()
    tabela = []
    for x in range(0, len(result)):
        mydata = list(filter(None, (result[x].strip()).split(" ")))
        if (mydata):
            tabela.append(mydata)
    return tabela


def column(matrix, i):
    return [float(row[i]) for row in matrix]


def graph(data):
    y = range(0, len(data))
    plt.plot(data)
    plt.plot(y, data, "ro")
    plt.xlabel('SAMPLE')
    plt.ylabel('DATA')
    plt.show()


def main():
    data = readArchive('archives/x01.txt')
    graph(data)

# main()
