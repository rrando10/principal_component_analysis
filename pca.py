#my imported libraries
import csv
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

#open and read data from .csv
data = pd.read_csv("UTK-peers-edit.csv",header=0)
data.head()

#make a data matrix w/ chosen numeric attributes
X = np.array(data,dtype = np.float64)

#compute svd of the data matrix
U, S, Vt = np.linalg.svd(X, full_matrices=True)
V = Vt.T

#create list of eigenvalues
eigvals = S**2 / np.cumsum(S)[-1]
sing_vals = np.arange(51) + 1

#plot scree graph
plt.plot(sing_vals, eigvals, 'bo-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')

leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3, 
                 shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                 markerscale=0.4)
leg.get_frame().set_alpha(0.4)
leg.draggable(state=True)

plt.show()

#create list of variances
var = S/100
var = np.square(var)

#reduce data matrix to 1st k PCs
PC1 = V[:,-51]
V = V[:,:-49]
PC2 = V[:,-1]

print(PC1)
print(PC2)

f, diagram = plt.subplots(1)

#scatter plot graph PC1 vs PC2
for i in range(len(V)): 
    x = V[i][0]
    y = V[i][1]
    diagram.plot(x,y,"bo")
    diagram.annotate(i, (x,y), fontsize=12)

plt.title('PC1 vs. PC2')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

