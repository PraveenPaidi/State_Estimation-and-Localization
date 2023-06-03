# State_Estimation-and-Localization

State Estimation:
State can be any number which is npostionor velocity or anything with just a number.
Correlation between the state parameters is captured by the correlation matrix and the this kind of covariance matrix helps in getting more information for the Kalman filter.

X^k=F_k X_(k-1)+B_k U_k
					  P_k=F_k P_(k-1) F_k^T+Q_k

In other words, the new best estimate is a prediction made from previous best estimate, plus a correction for known external influences.
And the new uncertainty is predicted from the old uncertainty, with some additional uncertainty from the environment.

https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/


