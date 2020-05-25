import numpy as np

# Warning : NOT an usual product of matrices
# Compute a column by column product taking in account NaN's

def transpose_A_times_B (X,Y) :
    
    d, N, M = X.shape[0], X.shape[1], Y.shape[1]
    XY = np.zeros((N,M))
    
    for i in range(0,N) :
        for j in range(0,M) :
            XY[i,j] = np.nanmean(X[:,i]*Y[:,j])*d
        print(i)
        
    return XY 
