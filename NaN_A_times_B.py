import numpy as np

# Do X times Y (Numpy arrays) taking into account NaN

def NaN_A_times_B(X, Y) :
    
    N = X.shape[1]
    X0, Y0 = X, Y
    u, v = np.isnan(X), np.isnan(Y) 
    X0[u], Y0[v] = 0, 0
    X1, Y1 = np.logical_not(u).astype(int), np.logical_not(v).astype(int)

    XY = np.dot(X0, Y0)
    NXY = np.dot(X1, Y1)
    XY = np.divide(np.dot(N,XY), NXY)
    
    return XY