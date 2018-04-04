# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 20:22:39 2018

@author: Natalia Spitha
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 20 


################################## Functions ##################################
def Gauss(t, a, t0, sigma):
    return a * np.exp(-(t - t0)**2 / (2 * sigma**2))

def bEqn(g, e, h, pulse, stepsize, cwrate=10000000, Gdecay=200, G2decay=7.4e-12, G3decay=1.4e-35, GHdecay=4.4e-12, Gescape=2.44e7, G3loss=1.08e-30, Gform=0):
    '''
    High concentration gem_pair master rate equation
    '''
    Gchange = (pulse
               + cwrate
               - g * Gdecay
               - g * g * G2decay
               - g * g * g * G3decay
               - g * h * GHdecay
               - g * Gescape
               - g * g * g *G3loss
               + e * h * Gform)
    return Gchange * stepsize


def hEqn(g, e, h, f, stepsize, Gescape=2.44e7, Gform=0, EHdecay=5.9e-13, FHloss=3.42e-12):
    """
    Hole master rate equation
    """
    hchange = (g * Gescape
               - h * e * EHdecay
               - h * f * FHloss
               - e * h * Gform)
    return hchange * stepsize


def eEqn(g, e, h, f, trap, stepsize, Gescape=2.44e7, Gform=0, EHdecay=5.9e-13, 
         Etrap=9.1e-13):
    """
    Electron master rate equation
    """
    echange = (g * Gescape
               - h * e * EHdecay
               - e * (trap - f) * Etrap
               - e * h * Gform)
    return echange * stepsize


def fEqn(e, h, f, trap, stepsize, Etrap=9.1e-13, FHloss=3.42e-12):
    """
    Filled trap master rate equation
    """
    fchange = (e * (trap - f) * Etrap
               - f * h * FHloss)
    return fchange * stepsize


def gsignalEqn(g, h, Gdecay=200, G2decay=7.4e-12, G3decay=1.4e-35, 
               GHdecay=4.4e-12):
    """
    G Recombination Signal
    """
    signal = (g * Gdecay
              + g * g * G2decay
              + g * g * g * G3decay
              + g * h * GHdecay)
    return signal


def ehsignalEqn(e, h, EHdecay=5.9e-13):
    """
    E-H Recombination Signal
    """
    signal = (h * e * EHdecay)
    return signal # Note that the scalar is not included here


def glossEqn(g, G3loss=1.08e-30):
    gloss = g * g * g * G3loss
    return gloss


def tlossEqn(f, h, FHloss=3.42e-12):
    '''
    Filled trap loss
    '''
    tloss = f * h * FHloss
    return tloss

############################### Definitions ###################################
fs = 1e-15
ps = 1e-12
ns = 1e-9
us = 1e-6
ms = 1e-3
MHz = 1e6
kHz = 1e3
nm = 1e-9
c = 2.9979e8
hPlanck = 6.626e-34

#Initial Parameters 
g = 0
e = 0
ho = 0
ft = 0
trap = 2.52e18 #cm^-3

reprate = 80*MHz
spacing = 1.00/reprate
N = 5000 #number of points in array corresponding to one cycle
numcycles = 20
duration = spacing*numcycles
stepsize = spacing/N
cwrate = 100e28 #carriers created per cm^3 per s

#Time and population arrays
time = np.arange(0,duration,stepsize)
gs = np.zeros(len(time))
es = np.zeros(len(time))
hos = np.zeros(len(time))
fts = np.zeros(len(time))
gsigs = np.zeros(len(time))
ehsigs = np.zeros(len(time))


def pulse(t, power=0.008*1e10, wavelength=400*nm, reprate=80*MHz, fwhm=100*fs):
    photonEnergy = hPlanck*c/wavelength
    pulseEnergy = power/reprate
    pulseCarriers = pulseEnergy/photonEnergy
    spacing = 1.00/reprate
    pulseSigma = fwhm/(2*np.sqrt(2*np.log(2)))
    pulsePeak = pulseCarriers/(np.sqrt(2*np.pi)*pulseSigma)
    return Gauss(t,pulsePeak,spacing/2,pulseSigma)
    
stime = np.linspace(0,spacing,N+1)[:-1]
spulse = pulse(stime)
stepsize = spacing/N

pulseTrain = np.tile(spulse,numcycles)

############################ Numerical Integration ############################
for i in np.arange(1,numcycles*N):
    gs[i] = gs[i-1] + bEqn(gs[i-1], es[i-1], hos[i-1], spulse[i%N], stepsize, cwrate)
    es[i] = es[i-1] + eEqn(gs[i-1], es[i-1], hos[i-1], fts[i-1], trap, stepsize)
    hos[i] = hos[i-1] + hEqn(gs[i-1], es[i-1], hos[i-1], fts[i-1], stepsize)
    fts[i] = fts[i-1] + fEqn(es[i-1], hos[i-1], fts[i-1], trap, stepsize)
    gsigs[i] = gsignalEqn(gs[i], hos[i])
    ehsigs[i] = ehsignalEqn(es[i], hos[i])

############################ Plotting #########################################
timens = time*1e9

#an unnecessarily fancy way to generate labels
if cwrate == 0:
    cwlabel = 'no CW'
else:
    expform = '{0:0.4e}'.format(cwrate)
    split = expform.split('e')
    nvalue = float(split[0])
    pow10 = int(split[1])
    cwlabel = '%0.2d'%nvalue + r'$\times 10^{' + str(pow10) + r'} \  cm^{-3} \ s^{-1}$'

#plt.subplot(311)
#plt.plot(timens, pulseTrain, label='pulse')
#plt.ylabel('Pulse Intensity (Carriers/s)')
#plt.subplot(211)
#plt.plot(timens, gs, label = 'G')
#plt.plot(timens, es, label = 'E')
#plt.plot(timens, hos, label = 'H')
#plt.plot(timens, fts, label = 'F')
#plt.ylabel('Population')
#plt.yscale('log')
#plt.legend()
#plt.subplot(212)
plt.plot(timens, gsigs+ehsigs, label = cwlabel)
plt.ylabel('PL Signal')
#plt.xlim(1,117.5)
#plt.yscale('log')
#plt.ylim(2e27,)
plt.legend()
plt.xlabel('Time (ns)')
plt.show()
