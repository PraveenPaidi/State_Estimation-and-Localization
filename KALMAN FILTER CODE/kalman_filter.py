import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import yaml

with open('/home/praveenpaidi/move_ws/file.yaml','r') as file:
    loaded=yaml.load(file,Loader=yaml.FullLoader)
pf=np.array(loaded['data'])
numOfMeasurements,n =np.shape(pf)

def filter(z, k,numOfMeasurements):
    if k<numOfMeasurements-1:
        dt = pf[k+1, 0] - pf[k,0]  
    
    # Initialize State
    if k == 0:
             
        filter.x = np.array([[0],[0]])            # state paramaters 
        filter.A = np.array([[1, dt],[0, 1]])     # state transition matrix 
        filter.H = np.array([[1, 0]])             # measurement matrix
        filter.HT = np.array([[1],[0]])           # transpose
        filter.Q = np.array([[0.05, 0],[0, 0.05]])     # covarinace error matrix
        filter.R = 3                               # measurement covarince matrix   it can be infintiy in the case of sensor irregularity
        filter.P = np.array([[1, 0],[0, 1]])       # covariance matrix   
        
    # In the case of R matrix is infinity the S matrix would go infinity as well. Then K = 0 for negligable sensor      
    # Predict State Forward
    x_p = filter.A.dot(filter.x)
    
    # Predict Covariance Forward
    P_p = filter.A.dot(filter.P).dot(filter.A.T) + filter.Q
    
    # Correction
    # Compute Kalman Gain
    
    S = filter.H.dot(P_p).dot(filter.HT) + filter.R
    K = P_p.dot(filter.HT)*(1/S)    # gain 
   
    # Estimate State
    
    
    residual = z - filter.H.dot(x_p)    # this is y
    filter.x = x_p + K*residual
    
    # Estimate Covariance
    filter.P = P_p - K.dot(filter.H).dot(P_p)
    
    
    return [filter.x[0]];

def testFilter():
    numOfMeasurements,n =np.shape(pf) 

    measTime = []
    measAcc = []
    
    for k in range(numOfMeasurements):
        z = [pf[k,1]]
        
        # Call Filter and return new State
        f = filter(z, k,numOfMeasurements)
        
        # Save off that state so that it could be plotted
        measTime.append(pf[k,0])
        measAcc.append(f[0])
        
    return [measTime, measAcc];

import scipy.signal as sg
nasdaq = pf[:,1]
shape=20
# We get a triangular window with 60 samples.
h = sg.get_window('triang', shape)
# We convolve the signal with this window.
fil = sg.convolve(nasdaq, h / h.sum(),mode='valid')
truncated= pf[0:numOfMeasurements-shape+1,0]

t = testFilter()

plot2 = plt.figure(1,figsize=(20,10))
plt.plot(t[0], t[1],'-k',label='Kalman filter Acceleration')
plt.plot(t[0], pf[:,1],'r',label='Raw Acceleration')
plt.plot(truncated, fil,'b',label='FIR filter Acceleration')
plt.ylabel('Linear Acceleration')
plt.xlabel('Time')
plt.title('Linear Acceleration x Estimate On 15 to 25 sec time step \n', fontweight="bold")
plt.xlim(15,25)
plt.legend()
plt.grid(True)

# plt.figure(2)
plot2 = plt.figure(2,figsize=(20,10))
plt.plot(t[0], t[1],'-k',label='Kalman filter Acceleration')
plt.plot(t[0], pf[:,1],'r',label='Raw Acceleration')
plt.plot(truncated, fil,'b',label='FIR filter Acceleration')
plt.ylabel('Linear Acceleration')
plt.xlabel('Time')
plt.title('Complete data Linear Acceleration x Estimate On Each Measurement Update \n', fontweight="bold")
plt.legend()
plt.grid(True)

