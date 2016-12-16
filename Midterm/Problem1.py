import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

dataset = genfromtxt('features.csv', delimiter = ' ')
y = dataset[:, 0]
X = dataset[:, 1:]
y[y<>0] = -1
y[y==0] = +1

#plots data
c0 = plt.scatter(X[y==-1,0], X[y==-1,1], s=20, color='r', marker='x')
c1 = plt.scatter(X[y==1,0], X[y==1,1], s=20, color='b', marker='o')

#displays legend
plt.legend((c0, c1), ('All Other Numbers -1', 'Number Zero +1'), loc='upper right', scatterpoints=1, fontsize=11)

#displays axis legends and title
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Intensity and Symmetry of Digits')

#saves the figure into a .pdf file (desired!)
plt.savefig('midterm.plot.pdf', bbox_inches='tight')
plt.show()
print X