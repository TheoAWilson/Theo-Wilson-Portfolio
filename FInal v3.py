#Imports===============================================================================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import subplot
from matplotlib import pyplot as plt 
from scipy.stats import mstats
from scipy.stats import mstats, gaussian_kde

#======================================================================================================================================
#Import data set

oring_dataset = {
    'Temperature': [53, 57, 58, 63, 66, 67, 67, 67, 68, 69, 70, 70, 70, 70, 72, 73, 75, 75, 76, 76, 78, 79, 81],
    'Failure Status': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
}

#Logistic Regression Function==========================================================================================================

def Logistic_reg(t,x_0,x_1):
    global prob_1
    prob_1 = 1/(1+(math.exp(-(x_0+(x_1*t)))))
    prob_1 = max(min(prob_1, 1 - 1e-10), 1e-10)
    #print(prob_1)
    return prob_1

#Generate Prior values=================================================================================================================
#mean then std
global std_0
global std_1
std_0 = math.sqrt(2.5)
std_1= math.sqrt(0.05)
#x0_start= np.random.normal(15, std_0)
#x1_start=np.random.normal(-0.25, std_1)
#std_0 = math.sqrt(3.16)
#std_1= math.sqrt(0.387)
x0_start= np.random.normal(15, std_0)
x1_start=np.random.normal(-0.22, std_1)

#Likelihood Function===================================================================================================================

def Lik_func(x_0, x_1):
    log_lik = 0
    temperatures = oring_dataset['Temperature']
    failure_status = oring_dataset['Failure Status']
    for i in range(len(temperatures)):
        prob = Logistic_reg(temperatures[i], x_0, x_1)
        #prob = max(min(prob, 1 - 1e-10), 1e-10)
        if failure_status[i] == 1:
            log_lik += math.log(prob)
        else:
            log_lik += math.log(1 - prob)
    return log_lik


#======================================================================================================================================
#Prob of norm
#enter std

def prob_norm0(value, mean):
    global prob_norm_value0
    prob_norm_value0=stats.norm.pdf(value,mean,std_0)
    prob_norm_value0 = math.log(prob_norm_value0)
    return prob_norm_value0

def prob_norm1(value, mean):
    global prob_norm_value1
    prob_norm_value1=stats.norm.pdf(value,mean,std_1)
    prob_norm_value1 = math.log(prob_norm_value1)
    return prob_norm_value1

#======================================================================================================================================
#HPD Func

def compute_hpd(sample, level=0.95):
    sample = np.sort(sample)
    n = len(sample)
    interval_increment = int(np.floor(level * n))
    intervals = sample[interval_increment:] - sample[:n - interval_increment]
    min_idx = np.argmin(intervals)
    return sample[min_idx], sample[min_idx + interval_increment]

#======================================================================================================================================
#Run MCMC


def MCMC(x0_cur,x1_cur):
    plt.figure(figsize=(12, 6))
    x1_values=[]
    x0_values=[]
    for i in range(1,50000):
        if i == 1:
            x0_old=x0_cur
            x1_old=x1_cur
        lik_cur=Lik_func(x0_cur,x1_cur)
        post_cur=(lik_cur)+(prob_norm0(x0_cur,x0_old))+(prob_norm1(x1_cur,x1_old))
        x1_new= np.random.normal(x1_cur, std_1)
        x0_new= np.random.normal(x0_cur, std_0)
        lik_new=Lik_func(x0_new,x1_new)
        post_new=(lik_new)+(prob_norm1(x1_new,x1_cur))+(prob_norm0(x0_new,x0_cur))


        #print("For the",[i],"th iteration the post new  is",post_new)
        #print("For the",[i],"th iteration the post cur  is",post_cur)

        #print("For the",[i],"th iteration the accept ratio  is",math.exp(post_new-post_cur))

        #print("For the",[i],"th iteration the Like_Cur  is",lik_cur)
        #print("For the",[i],"th iteration the Like_New  is",lik_new)

        #print("For the",[i],"th iteration thr prob of x1 is",prob_norm(x1_new,x1_cur))
        #print("For the",[i],"th iteration thr prob of x0 is",prob_norm(x0_new,x0_cur))

        ratio=min(1,(math.exp(post_new-post_cur)))
        U=random.uniform(0,1)
        if ratio >= U:
            x1_old=x1_cur
            x1_cur=x1_new
            x0_old=x0_cur
            x0_cur=x0_new
        else:
            x1_old=x1_cur
            x0_old=x0_cur
        print("For the",[i],"th iteration x0 value is",x0_cur,"x1 value is",x1_cur)


        if i > 5000:
            x1_values.append(x1_cur)
            x0_values.append(x0_cur)

    hpd_x0 = compute_hpd(x0_values, level=0.95)
    hpd_x1 = compute_hpd(x1_values, level=0.95)

    plt.clf()

    plt.subplot(2, 2, 1)
    #density_x0 = gaussian_kde(x0_values)
    #x_vals_x0 = np.linspace(min(x0_values), max(x0_values), 1000)
    plt.hist(x0_values, bins=100, color='black', density=True)
    #plt.plot(x_vals_x0, density_x0(x_vals_x0), color='red', linewidth=2)
    plt.axvline(hpd_x0[0], color='red', linestyle='dashed', label='HPD Lower')
    plt.axvline(hpd_x0[1], color='blue', linestyle='dashed', label='HPD Upper')
    plt.xlabel('beta0')
    plt.ylabel('Density')


    plt.subplot(2, 2, 2)
    plt.plot(x0_values, color='black', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('beta0')

    plt.subplot(2, 2, 3)
    #density_x1 = gaussian_kde(x1_values)
    x_vals_x1 = np.linspace(min(x1_values), max(x1_values), 1000)
    plt.hist(x1_values, bins=100, color='black', density=True)
    #plt.plot(x_vals_x1, density_x1(x_vals_x1), color='red', linewidth=2)
    plt.axvline(hpd_x1[0], color='red', linestyle='dashed', label='HPD Lower')
    plt.axvline(hpd_x1[1], color='blue', linestyle='dashed', label='HPD Upper') 
    plt.xlabel('beta1')
    plt.ylabel('Density')


    plt.subplot(2, 2, 4)
    plt.plot(x1_values, color='black', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('beta1')


    plt.show()
    print("x0 value is",x0_cur,"x1 value is",x1_cur)
    return x0_cur,x1_cur

MCMC(x0_start,x1_start)





