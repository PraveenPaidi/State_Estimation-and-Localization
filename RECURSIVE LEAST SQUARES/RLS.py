#!/usr/bin/env python
# coding: utf-8

# In[9]:


## sythetic dataset for the Recusrive Least squares theory

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

I = np.array([[0.2, 0.3, 0.4, 0.5, 0.6]]).T
V = np.array([[1.23, 1.38, 2.06, 2.47, 3.17]]).T


# In[10]:


## 
# ### Batch Estimator
# Before implementing recursive least squares, we review the parameter estimate given by the batch least squares method for comparison.
## Batch Solution
# compact solution adding ones for the 
H = np.ones((5,2))
H[:, 0] = I.reshape(5,)

# prediction values   least square solution
x_ls = inv(H.T.dot(H)).dot(H.T.dot(V))
print('The parameters of the line fit are ([R, b]):')
print(x_ls)

#Plot
I_line = np.arange(0, 0.8, 0.1)
V_line = x_ls[0]*I_line + x_ls[1]
plt.figure(2)
plt.scatter(np.asarray(I), np.asarray(V))
plt.plot(I_line, V_line, 'g')
plt.xlabel('current (A)')
plt.ylabel('voltage (V)')
plt.grid(True)
plt.show()


# In[11]:


# We begin with a prior estimate of R = 4.
# We assume perfect knowledge of current I, while voltage V data are corrupted by additive, independent and identically distributed Gaussian noise of variance 0.0255 V^2.
# R_hat ~ N(4,10) , b_hat ~ N (0,0.2)

# Initialize the parameter and covariance estimates: x0_hat = E[x], P0 = E[(x-x0_hat)*(x-x0_hat).T]
P_k = np.array([[10,0],[0,0.2]])   #covariance is choosen initially  
x_k = np.array([4, 0]).T           #compact notation X basically H in following code

#Our measurement variance
Var = 0.0225

#Pre allocate our solutions so we can save the estimate at every step
num_meas = I.shape[0]
x_hist = np.zeros((num_meas + 1,2))
P_hist = np.zeros((num_meas + 1,2,2))
print("P_hist.shape=",P_hist.shape)

x_hist[0] = x_k
P_hist[0] = P_k


# In[16]:


#Iterate over the measurements
for k in range(num_meas):
    
    #Construct the Jacobian H_k
    H_k = np.array([I[k], 1]).reshape(1, 2) 
    H_k = np.array(H_k, dtype=np.float64)
    R_k = np.array([Var])
      
    #Construct K_k - Gain Matrix   Kk = P(k-1)*Hk.T*(Hk*P(k-1)*Hk.T+Rk)^-1    
    K_k = P_hist[k].dot(H_k.T).dot(inv(H_k.dot(P_hist[k]).dot(H_k.T) + R_k))
    np.reshape(K_k, (2, 1))  
    
    #Update our estimate    xk_hat = x(k-1)_hat+Kk*(yk-Hk*x(k-1)_hat)
    x_k = x_hist[k].reshape(2, 1) + K_k.dot(V[k] - H_k.dot(x_hist[k].reshape(2, 1)))
   
    #Update our uncertainty - Estimator Covariance
    P_k = (np.eye(2, dtype=int) - K_k.dot(H_k)).dot(P_hist[k])
       
    #Keep track of our history
    P_hist[k+1] = P_k.reshape(2, 2)
    x_hist[k+1] = x_k.reshape(1, 2)
    
print('The parameters of the line fit are ([R, b]):')
print(x_k)

# Plot results
I_line = np.arange(0, 0.8, 0.1)
plt.figure(3)
plt.scatter(np.asarray(I), np.asarray(V))
plt.xlabel('current (A)')
plt.ylabel('voltage (V)')
plt.plot(I_line, V_line, label='Batch Least-Squares Solution')
for k in range(num_meas):
    V_line = x_hist[k,0]*I_line + x_hist[k,1]
    plt.plot(I_line, V_line, label='Measurement {}'.format(k))

plt.grid(True)
plt.legend()
plt.show()


# In[ ]:




