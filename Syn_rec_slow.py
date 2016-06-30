#/usr/bin/python

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 02 15:59:18 2015

@author: cmi
"""

from __future__ import division
import numpy as np
from scipy.optimize import fmin, fmin_powell
from scipy.integrate import odeint
import matplotlib.pyplot as plt

gl = globals().copy()
for var in gl:
    if var[0] == '_': continue
    if 'func' in str(globals()[var]): continue
    if 'module' in str(globals()[var]): continue
    del globals()[var]

# This function resolve the ODE part of the biomehanical model (the synaptic events make the differential equation not linear)
# The full model is a modified version of the Tsodiks-Markram synapse model, we added a slow and a very slow component 
# to match os close as possible the synapse physiology 
def ODEfunc(x,t):
    
    global tr, tf, ti, ts, p0
    
    # here is the neurotransmitter vesicle pool replenishment
    dx0 = (1-x[0])/tr
    # here is the release facilitation
    dx1 = (x[2]-x[1])/tf
    # slow component (calcium channel inactivation)
    dx2 = (x[3]-x[2])/ti
    # very slow component (activation of G-proteins for instance by presynaptic mGluR)
    dx3 = (p0-x[3])/ts
    
    return [dx0, dx1, dx2, dx3]

def mod(t):
    
    # that two part model the synapse stimulation protocol: the first one is the stimulation allowing the measurment 
    # of the initial and late depression, the second one is the protocol of recovery measurement
    
    global time, kf, ki, ks, p0, tr
        
    init = np.array([1,p0,p0,p0])
    S = np.zeros((int(sum(Stim))+6,len(init)))
    time = t[0:int(round(1/F/Te))]
    for i in range(int(sum(Stim))):
                
        Y = odeint(ODEfunc,init,time)

        S[i,:] = [Y[-1,0], Y[-1,1], Y[-1,2], Y[-1,3]]
        
        # the following are the amount  of event driven synapse modification 
        # release of transmitter
        init[0] = S[i,0] - S[i,0]*S[i,2]
        # facilitation
        init[1] = S[i,1] + kf*(1-S[i,1])
        # calcium channel inactivation
        init[2] = S[i,2] - ki*S[i,2]
        # mGluR activation
        init[3] = S[i,3] - ks*S[i,3]
        
    Int = tr
    tr = 8    
    # the following tab is the index of recovery protocol stimulation
    IndRec = [100, 1000, 2000, 5000, 10000, 20000]
    for j in range(6):
        
        time = t[0:int(round(IndRec[j]/Te))]
        Y = odeint(ODEfunc,init,time)
        
        S[i+j+1,:] = [Y[-1,0], Y[-1,1], Y[-1,2], Y[-1,3]]
        
        init[0] = S[i+j+1,0] - S[i+j+1,0]*S[i+j+1,2]
        init[1] = S[i+j+1,1] + kf*(1-S[i+j+1,1])
        init[2] = S[i+j+1,2] - ki*S[i+j+1,2]
        init[3] = S[i+j+1,3] - ks*S[i+j+1,3]
        
    tr = Int
        
    return S

# that is the function of optimisation of the model parameters against the date (when present)
def func(P, t, y):
    
    # the model parameters have to be global to be disponible for the model
    global tr, tf, kf, p0, ti, ki, ts, ks, LB, HB, time
            
    tr = P[0]
    tf = P[1]
    kf = P[2]
    p0 = P[3]
    ti = P[4]
    ki = P[5]
    ts = P[6]
    ks = P[7]
    
    # with the new set of paramters the synapse is simulated
    M = mod(t)

    Syn = M[:,0]*M[:,2]
    Syn = Syn/Syn[0]
    
    # the following allows the user to impose a physiological range of parameters, for which, if outside, the cost function
    # will be pohibitive
    A = 0
    for a in np.arange(len(P)):
        if P[a] > HB[a] :
            A = A + abs(P[a] - HB[a])*1e6
        if P[a] < LB[a] :
            A = A + abs(LB[a]-P[a])*1e6
            
    return sum(pow(np.array(y - Syn),2)) + A


duration = 7
TeDat = 0.01
tDat = np.arange(0,duration,TeDat)
tDat = np.concatenate((tDat,[tDat[-1]+0.1]),axis=0)
tDat = np.concatenate((tDat,[tDat[-2]+1]),axis=0)
tDat = np.concatenate((tDat,[tDat[-3]+2]),axis=0)
tDat = np.concatenate((tDat,[tDat[-4]+5]),axis=0)
tDat = np.concatenate((tDat,[tDat[-5]+10]),axis=0)
tDat = np.concatenate((tDat,[tDat[-6]+20]),axis=0)

Te = 0.001
t = np.arange(0,duration+Te,Te)
t2 = np.arange(0,50+Te,Te)

F = 100
Stim = np.zeros(len(t))
Stim[int(round(1/F/Te))-1::int(round(1/F/Te))] = 1

time = t[0:int(round(1/F/Te))]
   
[     tr,      tf,     kf,     p0,     ti,   ki,      ts,    ks] = \
    [ 0.08,     0.04,   0.07,   0.05,   5,    0.005,   30,    0.005]
        
Y0 = [tr, tf, kf, p0, ti, ki, ts, ks]

IC = Y0

LB = [0.05,     0.01,   0.01,   0.01,   1,    0.001,    10,    0.001]
HB = [0.20,     0.1,    0.1,    0.4,    10,   0.0015,   100,   0.01]

M = mod(t)

Syn = M[:,0]*M[:,2]
Syn = Syn/Syn[0]
    
plt.figure(facecolor=[1,1,1])

plt.plot(tDat+TeDat-Te,Syn,'r')
plt.plot(tDat+TeDat-Te,M)

plt.legend(('syn','n','p','slow'),fontsize=15,loc=4)  

plt.ylim([-0.1,1.1])

plt.xlabel('time (s)')

plt.show()
