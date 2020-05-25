import pyreadr
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

os.getcwd() 
os.chdir('/Users/joshuawolff/Desktop/PRE/Codes R température')


#######################
# Import of the dataset
#######################

training_data = pyreadr.read_r('DATA_TRAINING.RData')

print(training_data.keys())
loc = training_data['loc']
time, year, month, day =training_data['time'], training_data['year'], training_data['month'], training_data['day']
Adf = training_data['anom.training'] # Training dataset
index_training = training_data['index.training']
index_validation = training_data['index.validation']



###############################################
# Computes and removes the trend of the anomaly
###############################################

T = len(time) # number of instants considered 
n = int(len(Adf)/T) # Number of nodes in the grid
t = [i for i in range(1, T+1)] # time vector

# A must be a T x n matrix (it's a vector for the moment)
A = Adf.values.reshape(T, n)

# tyear contains an instant in each year, we will use it to simplify the regression
# on the trend
tyear = pd.DataFrame({'year':np.ndarray.flatten(year.values), 't':t}).groupby('year').mean()



from sklearn.linear_model import LinearRegression

compute_Mu = 1 # 0 if Mu has been calculated, 1 if not
if compute_Mu == 0 :
     Mu = np.load('Mu.npy')
else :
    # polynomial regression : trend = a + b*t**2 + c*t
    # the regression is made with tyear (annual mean of t) to simplify the computation
    t2 = np.power(t, 2)
    tyear2 = np.power(tyear, 2) 
    Mu = np.zeros((T, n))
    B = np.zeros((n, 3))
    for i in range(0,n) :
        a_i = A[:,i]
        ayear_i = pd.DataFrame({'year':np.ndarray.flatten(year.values), 'a_i':a_i}).groupby('year').mean()
        fit = LinearRegression().fit(pd.merge(tyear2, tyear, on = 'year'), ayear_i)
        b_i=np.concatenate((fit.intercept_,fit.coef_[0]))
        Mu[:,i] = b_i[0]+np.dot(b_i[1],t2)+np.dot(b_i[2],t) # Mu contains the trend for each spation-temporal point
        B[i,:] = b_i # contains the coefficients of the regression (intercept,squared,linear)
    print('Regression Done')
    np.save('Mu.npy', Mu) 
    print("Mu saved")

A_raw = A
print('Removes the trend')
A = A_raw - Mu # We remove the trend of the anomaly
