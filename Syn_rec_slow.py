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

plt.close('all')

gl = globals().copy()
for var in gl:
    if var[0] == '_': continue
    if 'func' in str(globals()[var]): continue
    if 'module' in str(globals()[var]): continue
    del globals()[var]


def ODEfunc(x,t):
    
    global tr0, tf, ti, p0
    
    dx0 = (1-x[0])/tr0
    dx1 = (x[2]-x[1])/tf
    dx2 = (p0-x[2])/ti
    
    return [dx0, dx1, dx2]

def mod(t):
    
    global time, kf, ki, p0, tr0
        
    init = np.array([1,p0,p0])
    S = np.zeros((int(sum(Stim))+6,len(init)))
    time = t[0:int(round(1/F/Te))]
    for i in range(int(sum(Stim))):
                
        Y = odeint(ODEfunc,init,time)

        S[i,:] = [Y[-1,0], Y[-1,1], Y[-1,2]]
        
        init[0] = S[i,0] - S[i,0]*S[i,2]
        init[1] = S[i,1] + kf*(1-S[i,1])
        init[2] = S[i,2] - ki*S[i,2]

    Int = tr0
    tr0 = 8    
    IndRec = [100, 1000, 2000, 5000, 10000, 20000]
    for j in range(6):
        
        time = t[0:int(round(IndRec[j]/Te))]
        Y = odeint(ODEfunc,init,time)
        
        S[i+j+1,:] = [Y[-1,0], Y[-1,1], Y[-1,2]]
        
        init[0] = S[i+j+1,0] - S[i+j+1,0]*S[i+j+1,2]
        init[1] = S[i+j+1,1] + kf*(1-S[i+j+1,1])
        init[2] = S[i+j+1,2] - ki*S[i+j+1,2]
    tr0 = Int
        
    return S


def func(P, t, y):
    
    global tr0, tf, kf, p0, ti, ki, LB, HB, time
            
    tr0 = P[0]
    tf = P[1]
    kf = P[2]
    p0 = P[3]
    ti = P[4]
    ki = P[5]
    
    M = mod(t)

    Syn = M[:,0]*M[:,2]
    Syn = Syn/Syn[0]
    
    A = 0
    for a in np.arange(len(P)):
        if P[a] > HB[a] :
            A = A + abs(P[a] - HB[a])*1e6
        if P[a] < LB[a] :
            A = A + abs(LB[a]-P[a])*1e6
            
    return 20*sum(pow(np.array(y[0:50] - Syn[0:50]),2)) \
    + sum(pow(np.array(y[50:-6] - Syn[50:-6]),2)) \
    + 10*sum(pow(np.array(y[-6:] - Syn[-6:]),2)) + A


NbFiles = 6

A = np.zeros((1032258,NbFiles))

t,A[:,0] = np.loadtxt('2015_02_02_0012.txt')
t,A[:,1] = np.loadtxt('2015_02_02_0013.txt')
t,A[:,2] = np.loadtxt('2015_02_02_0014.txt')
t,A[:,3] = np.loadtxt('2015_02_02_0015.txt')
t,A[:,4] = np.loadtxt('2015_02_02_0016.txt')
t,A[:,5] = np.loadtxt('2015_02_02_0017.txt')

tInit = t

A = A-np.mean(A[0:500,:],axis=0)

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


params = np.zeros((NbFiles,7))
for k in np.arange(0,NbFiles):
    
    data = A[:,k]
    
    plt.figure(0,facecolor=[1,1,1])
    plt.subplot(211)
    plt.plot(tInit,data)
                
    TeInit = tInit[1]-tInit[0]
    data = np.array(data)
    
    data = data/np.amin(data[16500:16700],axis=0)
    
    plt.subplot(212)
    plt.plot(tInit,data)
    
    peakind = np.zeros(int(duration/TeDat+6))
    for j in range(int(duration/TeDat)):
        Ind = np.argmax(data[int(0.825/TeInit+j*TeDat/TeInit):int(0.825/TeInit+(j+1)*TeDat/TeInit)])
        peakind[j] = 0.825/TeInit + j*TeDat/TeInit + Ind
        
    l = j    
    IndRec = [0.1, 1, 2, 5, 10, 19.98]
    m = 0
    for j in range(l+1,l+1+6):
        Ind = np.argmax(data[int(0.815/TeInit+30/TeInit+IndRec[m]/TeInit):int(0.815/TeInit+30/TeInit+IndRec[m]/TeInit+200)])
        peakind[j] = 0.815/TeInit + 30/TeInit + IndRec[m]/TeInit + Ind
        m = m + 1
        
    plt.plot(tInit[peakind.astype(int)],data[peakind.astype(int)],'r*')
    
    data = data[peakind.astype(int)]
#    for j in np.arange(-6,0): 
#        tDat = np.concatenate((tDat,[tInit[peakind[j].astype(int)]-(30-duration+0.825)]), axis=0)
    plt.figure(2,facecolor=[1,1,1])
    plt.subplot(2,3,k+1)
    plt.plot(tDat+TeDat-TeInit,data,'k')
    
    [     tr0,      tf,     kf,     p0,     ti,   ki] = \
        [ 0.05,     0.05,   0.04,   0.05,   5,   0.0008]
            
    Y0 = [tr0, tf, kf, p0, ti, ki]
    
    IC = Y0
    
    LB = [0.01,    0.01,   0.01,   0.01,   1,    0.0001]
    HB = [0.1,     0.1,    0.1,    0.4,    50,   0.0015]
    
    
    #    Y2 = fmin_powell(func, Y0, args=(t, data),maxfun=100)
    Y0 = fmin(func, Y0, args=(t, data),maxfun=5000)
    [tr0, tf, kf, p0, ti, ki] = Y0
       
    M = mod(t)
    
    Syn = M[:,0]*M[:,2]
    Syn = Syn/Syn[0]
    
    GoF = sum(pow(np.array(data - Syn),2))
    
    params[k,:] = [tr0, tf, kf, p0, ti, ki, GoF]
        
    plt.figure(2)
    
    plt.rc('font', size=10)
    
    plt.plot(tDat+TeDat-Te,Syn,'r')
    plt.plot(tDat+TeDat-Te,M)

    if k == 0:
        plt.legend(('data','syn','n','p','slow'),fontsize=8,loc=4)  
    
    #plt.xlim([0,0.8])
    plt.ylim([-0.1,1.1])
    
    plt.xlabel('time (s)')

    k = k+1
    
    plt.draw()

print 'initial conditions :'
print IC
    
print '[[tr0, tf, kf, p0, ti, ki]] = '
print params[:,:-1]

print 'GoF = '
print params[:,-1]

print 'Total GoF = ',sum(params[:,-1])

plt.savefig('fig_simple.png')

np.savetxt('simple.txt',params)