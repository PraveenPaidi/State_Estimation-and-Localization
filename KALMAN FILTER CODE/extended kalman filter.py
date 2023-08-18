import numpy as np
import scipy as sp
import math 
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
        filter.H = np.array([[1, 0]])             # measurement matrix
        filter.Q = np.array([[0.1, 0.0], [0.0, 0.1]])  # Process noise covariance
        filter.R = np.array([[1.0]])  # Measurement noise covariance 
        filter.P = np.array([[1, 0],[0, 1]])       # covariance matrix 

        filter.HT = np.array([[1],[0]])           # transpose
         

    # State transition function with a cosine influence
    def f(x):
        position, velocity = x
        new_position = position + velocity * dt
        new_velocity = velocity + 0.1 * math.cos(position)  # Incorporating a cosine influence
        return np.array([new_position, new_velocity])
    

    # non linear functions are used to measure the mean but not the variance, locally linear approximations are used for the variance.   
    # Predict State Forward      B matrix and the noise are 0 here.
    x_p = f(filter.x)   # for Extended this A matrix becomes some other thing w.r.t to dA/ dx (k-1)
    # Generally A is obtained by the jacobian of the x matrix.   by the previous measurment as it is most close data.
    
    filter.A = np.array([[1.0, dt], [-0.1 * dt * math.sin(x_p[0]), 1.0]])  # Jacobian of f
    # Predict Covariance Forward
    P_p = filter.A.dot(filter.P).dot(filter.A.T) + filter.Q
    
    # Correction
    # Compute Kalman Gain


    
    S = filter.H.dot(P_p).dot(filter.HT) + filter.R      # here H matrix is also mapping between the original and the 
    #measurmenmt and the prediction, if it is a function we have take jacobian of it as well   dh/ dxk of the current prediction
    
    K = P_p.dot(filter.HT)*(1/S)    # gain 
   
    # Estimate State    
    residual = z - filter.HT.dot(x_p)    # this is y

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

