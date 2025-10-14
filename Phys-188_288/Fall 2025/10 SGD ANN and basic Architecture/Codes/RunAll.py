# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 00:31:36 2025

@author: MMH_user
"""

nclasses = 5

[x, y]         = spiral_data(samples = 200, classes = nclasses)
[x_new, y_new] = spiral_data(samples = 50, classes = nclasses)

E = [10, 100, 1000, 10000]

Color = np.random.uniform(0,1, (nclasses,3))

for e in E:
    RunMyANN(x, y, Nepoch = e, plot = False)
    #[x, y] = spiral_data(samples = e, classes = nclasses)
    
    #RunMyANN(x, y, Nepoch = 10000, plot = False)
    [predictions, probabilities] = ApplyMyANN(x_new)
    
    for i, cl in enumerate(range(nclasses)):
        idx  = np.argwhere(y_new == cl)
        idxp = np.argwhere(predictions == cl)
        
        color = Color[i,:]
        
        plt.scatter(x_new[idx,0], x_new[idx,1], facecolors = color)
        plt.scatter(x_new[idxp,0], x_new[idxp,1], facecolors = 'none', edgecolors = color)
    plt.show()
    

