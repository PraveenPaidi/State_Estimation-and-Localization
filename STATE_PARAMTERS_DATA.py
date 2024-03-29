#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion

# For parts 1 and 2, you will use p1_data.pkl. For Part 3, you will use pt3_data.pkl.
#DATA
with open('data/pt1_data.pkl', 'rb') as file:
    data = pickle.load(file)


# In[59]:


gt = data['gt']  # ground truth values
imu_f = data['imu_f']
imu_w = data['imu_w']
gnss = data['gnss']
lidar = data['lidar']


# In[60]:


#   gt: Data object containing ground truth. with the following fields:
#     a: Acceleration of the vehicle, in the inertial frame
#     v: Velocity of the vehicle, in the inertial frame
#     p: Position of the vehicle, in the inertial frame
#     alpha: Rotational acceleration of the vehicle, in the inertial frame
#     w: Rotational velocity of the vehicle, in the inertial frame
#     r: Rotational position of the vehicle, in Euler (XYZ) angles in the inertial frame
#     _t: Timestamp in ms.


#   imu_f: StampedData object with the imu specific force data (given in vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.


#   imu_w: StampedData object with the imu rotational velocity (given in the vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.


#   gnss: StampedData object with the GNSS data.
#     data: The actual data
#     t: Timestamps in ms.



#   lidar: StampedData object with the LIDAR data (positions only).
#     data: The actual data
#     t: Timestamps in ms.


# In[61]:


################################################################################################
# Let's plot the ground truth trajectory to see what it looks like. When you're testing your
# code later, feel free to comment this out.
################################################################################################
gt_fig = plt.figure()
ax = gt_fig.add_subplot(111, projection='3d')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Ground Truth trajectory')
ax.set_zlim(-1, 5)
plt.show()


# In[62]:


print("LIDAR")
print(lidar.data.shape)
print(lidar.t.shape)
print("IMU_F")
print(imu_f.data.shape)
print(imu_f.t.shape)
print("IMU_W")
print(imu_w.data.shape)
print(imu_w.t.shape)
print("GNSS_DATA")
print(gnss.data.shape)
print(gnss.t.shape)
print("GT")
print(gt.a.shape)
print(gt.v.shape)
print(gt.p.shape)
print(gt.alpha.shape)
print(gt.w.shape)
print(gt.r.shape)
print(gt._t.shape)


# In[63]:


C_li = np.array([
   [ 0.99376, -0.09722,  0.05466],
   [ 0.09971,  0.99401, -0.04475],
   [-0.04998,  0.04992,  0.9975 ]
])

t_i_li = np.array([0.5, 0.1, 0.5])

# Transform from the LIDAR frame to the vehicle (IMU) frame. as we are finding the positions in IMU frame.
lidar.data = (C_li @ lidar.data.T).T + t_i_li


# In[64]:


# constants
var_imu_f = 0.10
var_imu_w = 0.25
var_gnss  = 0.01
var_lidar = 1.00

# initialization 

p_est = np.zeros([imu_f.data.shape[0], 3])  # position estimates
v_est = np.zeros([imu_f.data.shape[0], 3])  # velocity estimates
q_est = np.zeros([imu_f.data.shape[0], 4])  # orientation estimates as quaternions
p_cov = np.zeros([imu_f.data.shape[0], 9, 9])  # covariance matrices at each timestep


# In[65]:


# Set initial values.
p_est[0] = gt.p[0]      # 3 values of position 
v_est[0] = gt.v[0]      # 3 values of velocity
q_est[0] = Quaternion(euler=gt.r[0]).to_numpy()    # converting euler angles to quarternian
p_cov[0] = np.zeros(9)  # covariance of estimate
gnss_i  = 0
lidar_i = 0


# In[66]:


g = np.array([0, 0, -9.81])  # gravity
l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3)  # measurement model jacobian


# In[67]:


def measurement_update(sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
    # 3.1 Compute Kalman Gain # C2M5L2P13
    R = sensor_var * np.eye(3)
    K_k = p_cov_check @ h_jac.T @ np.linalg.inv(h_jac @ p_cov_check @ h_jac.T + R)  

    # 3.2 Compute error state # C2M5L2P14
    error_state = np.dot(K_k, (y_k - p_check))

    # 3.3 Correct predicted state # C2M5L2P15
    delta_p = error_state[0:3]
    delta_v = error_state[3:6]
    delta_phi = error_state[6:9]

    p_hat = p_check + delta_p
    v_hat = v_check + delta_v
    q_hat = Quaternion(axis_angle=delta_phi).quat_mult_right(q_check, out='np')

    # 3.4 Compute corrected covariance_check
    p_cov_hat = (np.eye(9) - K_k @ h_jac) @ p_cov_check

    return p_hat, v_hat, q_hat, p_cov_hat


# In[68]:


for k in range(1, imu_f.data.shape[0]):  # start at 1 b/c we have initial prediction from gt
    delta_t = imu_f.t[k] - imu_f.t[k - 1]    # time step for the each iteration can be constant even
    
    # calculation of rotation matrix from quoternion
    C_ns = Quaternion(*q_est[k - 1]).to_mat()
    
    # motion model
    p_est[k] = p_est[k - 1] + delta_t * v_est[k - 1] +(delta_t ** 2 / 2) * (C_ns @ imu_f.data[k - 1] + g)#s = s(n-1) +ut +0.5at^2
    v_est[k] = v_est[k - 1] + delta_t * (C_ns @ imu_f.data[k - 1] + g)                                   #v= u+ at
    # quaternian update is multiplication of quarternion as the rotation multiplication gives the update of the rotation.
    # taking angle axis roation inside the quoternian and then multiplying 
    q_est[k] = Quaternion(axis_angle=imu_w.data[k - 1] * delta_t).quat_mult_right(q_est[k - 1]) 
    
    # Its not linear yet, it has to linerized
    
    #### Making the F matrix by combining the aboev p , v and rotation update into single one of matrix form
    F_k = np.eye(9)
    F_k[0:3, 3:6] = delta_t * np.eye(3)
    F_k[3:6, 6:9] = -(C_ns.dot(skew_symmetric(imu_f.data[k - 1].reshape((3, 1))))) * delta_t
    
    Q_k = np.eye(6)
    Q_k[0:3, 0:3] *= delta_t ** 2 * var_imu_f
    Q_k[3:6, 3:6] *= delta_t ** 2 * var_imu_w
    
    
    # 2. Propagate uncertainty
    p_cov[k] = F_k @ p_cov[k - 1] @ F_k.T + l_jac @ Q_k @ l_jac.T
    
    # 3. Check availability of GNSS and LIDAR measurements
    if gnss_i < gnss.t.shape[0]:
        if gnss.t[gnss_i] == imu_f.t[k - 1]:
            p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(var_gnss, p_cov[k], gnss.data[gnss_i].T,
                                                                        p_est[k], v_est[k], q_est[k])
            gnss_i += 1

    if lidar_i < lidar.t.shape[0]:
        if lidar.t[lidar_i] == imu_f.t[k - 1]:
            p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(var_lidar, p_cov[k], lidar.data[lidar_i].T,
                                                                        p_est[k], v_est[k], q_est[k])
            lidar_i += 1
    
    
    
    
    


# In[73]:


print(gt.p.shape)
print(p_est.shape)


# In[71]:


est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111, projection='3d')
ax.plot(p_est[:,0], p_est[:,1], p_est[:,2], label='Estimated')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2], label='Ground Truth')
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_zlabel('Up [m]')
ax.set_title('Ground Truth and Estimated Trajectory')
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
ax.set_zlim(-2, 2)
ax.set_xticks([0, 50, 100, 150, 200])
ax.set_yticks([0, 50, 100, 150, 200])
ax.set_zticks([-2, -1, 0, 1, 2])
ax.legend(loc=(0.62,0.77))
ax.view_init(elev=45, azim=-50)
plt.show()


# In[ ]:


################################################################################################
# We can also plot the error for each of the 6 DOF, with estimates for our uncertainty
# included. The error estimates are in blue, and the uncertainty bounds are red and dashed.
# The uncertainty bounds are +/- 3 standard deviations based on our uncertainty (covariance).
################################################################################################
error_fig, ax = plt.subplots(2, 3)
error_fig.suptitle('Error Plots')
num_gt = gt.p.shape[0]
p_est_euler = []
p_cov_euler_std = []

# Convert estimated quaternions to euler angles
for i in range(len(q_est)):
    qc = Quaternion(*q_est[i, :])
    p_est_euler.append(qc.to_euler())

    # First-order approximation of RPY covariance
    J = rpy_jacobian_axis_angle(qc.to_axis_angle())
    p_cov_euler_std.append(np.sqrt(np.diagonal(J @ p_cov[i, 6:, 6:] @ J.T)))

p_est_euler = np.array(p_est_euler)
p_cov_euler_std = np.array(p_cov_euler_std)

# Get uncertainty estimates from P matrix
p_cov_std = np.sqrt(np.diagonal(p_cov[:, :6, :6], axis1=1, axis2=2))

titles = ['Easting', 'Northing', 'Up', 'Roll', 'Pitch', 'Yaw']
for i in range(3):
    ax[0, i].plot(range(num_gt), gt.p[:, i] - p_est[:num_gt, i])
    ax[0, i].plot(range(num_gt),  3 * p_cov_std[:num_gt, i], 'r--')
    ax[0, i].plot(range(num_gt), -3 * p_cov_std[:num_gt, i], 'r--')
    ax[0, i].set_title(titles[i])
ax[0,0].set_ylabel('Meters')

for i in range(3):
    ax[1, i].plot(range(num_gt),         angle_normalize(gt.r[:, i] - p_est_euler[:num_gt, i]))
    ax[1, i].plot(range(num_gt),  3 * p_cov_euler_std[:num_gt, i], 'r--')
    ax[1, i].plot(range(num_gt), -3 * p_cov_euler_std[:num_gt, i], 'r--')
    ax[1, i].set_title(titles[i+3])
ax[1,0].set_ylabel('Radians')
plt.show()

