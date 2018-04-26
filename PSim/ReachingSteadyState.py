# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 20:22:39 2018


@author: Natalia Spitha
"""

#-----Imports / matplotlib params ----------------------------------------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.size'] = 25
matplotlib.rcParams['lines.linewidth'] = 3

#----- Function-independent Definitions ----------------------------------------

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

#time parameters & default values
#cwrate = 0
pulsePower = 8e7
reprate = 80*MHz
spacing = 1.00/reprate
N = 5000 #number of points in array corresponding to one cycle
numcycles = 20
duration = spacing*numcycles
stepsize = spacing/N
time = np.arange(0,duration,stepsize)

#----- Functions ---------------------------------------------------------------


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


def pulseP(power=pulsePower, wavelength=400*nm, reprate=80*MHz, fwhm=100*fs):
    '''
    Returns peak injected carrier density from a pulse given its power,
    wavelength, rep rate, and FWHM
    '''
    photonEnergy = hPlanck*c/wavelength
    pulseEnergy = power/reprate
    pulseCarriers = pulseEnergy/photonEnergy
    pulseSigma = fwhm/(2*np.sqrt(2*np.log(2)))
    pulsePeak = pulseCarriers/(np.sqrt(2*np.pi)*pulseSigma)
    return pulsePeak

def pulse(t, pPeak=pulseP(pulsePower), reprate=80*MHz, fwhm=100*fs):
    '''
    produces the pulse (over a time range of 1/reprate) given its peak power 
    and Gaussian parameters
    '''
    pulseSigma = fwhm/(2*np.sqrt(2*np.log(2)))
    spacing = 1.00/reprate
    return Gauss(t,pPeak,spacing/2,pulseSigma)

#----- Function-dependent Definitions ------------------------------------------

#Production of pulse and pulse train
pulsePeak = pulseP(power = pulsePower)
print("pulse peak", pulsePeak)
stime = np.linspace(0,spacing,N+1)[:-1]
spulse = pulse(stime)
stepsize = spacing/N
pulseTrain = np.tile(spulse,numcycles)

#Initial Parameters 
g = 0
e = 0
ho = 0
ft = 0
trap = 2.52e18 #cm^-3

#----- Numerical Integration ---------------------------------------------------

def integration(): 
    #initialize population arrays
    gs = np.zeros(len(time))
    es = np.zeros(len(time))
    hos = np.zeros(len(time))
    fts = np.zeros(len(time))
    gsigs = np.zeros(len(time))
    ehsigs = np.zeros(len(time))

    for i in np.arange(1,numcycles*N):
        gs[i] = gs[i-1] + bEqn(gs[i-1], es[i-1], hos[i-1], spulse[i%N], stepsize, cwrate)
        es[i] = es[i-1] + eEqn(gs[i-1], es[i-1], hos[i-1], fts[i-1], trap, stepsize)
        hos[i] = hos[i-1] + hEqn(gs[i-1], es[i-1], hos[i-1], fts[i-1], stepsize)
        fts[i] = fts[i-1] + fEqn(es[i-1], hos[i-1], fts[i-1], trap, stepsize)
        gsigs[i] = gsignalEqn(gs[i], hos[i])
        ehsigs[i] = ehsignalEqn(es[i], hos[i])
    
    return gs, es, hos, fts, gsigs, ehsigs

gs, es, hos, fts, gsigs, ehsigs = integration()

#----- Labels ------------------------------------------------------------------
def cwlabels(cwrate):
    if cwrate == 0:
        cwlabel = 'No CW Intensity'
    else:
        expform = '{0:0.4e}'.format(cwrate)
        split = expform.split('e')
        nvalue = float(split[0])
        pow10 = int(split[1])
        cwlabel = '%0.2d'%nvalue + r'$\times 10^{' + str(pow10) + r'} \  cm^{-3} \ s^{-1}$ from CW'
    return cwlabel

def pplabels():    
    if pulsePower == 0:
        pplabel = 'no pulse'
    else:
        expformp = '{0:0.4e}'.format(pulsePeak)
        splitp = expformp.split('e')
        nvaluep = float(splitp[0])
        pow10p = int(splitp[1])
        pplabel = '%0.2d'%nvaluep + r'$\times 10^{' + str(pow10p) + r'} \  cm^{-3} \ s^{-1}$'

#----- Plotting ---------------------------------------------------------------
timens = time*1e9

#--- Show evolution of carrier populations while reaching steady state --------
if 0:
    cwrate = 0
    gs, es, hos, fts, gsigs, ehsigs = integration()
    lalpha = 0.6 #transparency lv for lines
    palpha = 0.1 #transparency lv for patches
    
    plt.figure() 
    plt.subplot(211)
    plt.title("Evolution of Carrier Populations")
    plt.axvspan(6.1,6.1+12.5,alpha=palpha, color = 'yellow')
    plt.axvspan(231.1,231.1+12.5,alpha=palpha, color = 'teal')
    plt.plot(timens,gs,label='G', alpha=lalpha); plt.plot(timens,es,label='E', alpha=lalpha)
    plt.plot(timens,hos,label='H', alpha=lalpha); plt.plot(timens,fts,label='F', alpha=lalpha)
    plt.grid()
    plt.legend()
    plt.xlabel("Time (ns)"); plt.ylabel(r'Carrier Density (cm$^{-3}$)')
    plt.ylim(1e15,6e19)
    plt.yscale('log')
    
    plt.subplot(223)
    plt.title("Initial Pulse")
    tstart = 6.1
    plt.axvspan(tstart,tstart+12.5,alpha=palpha, color = 'yellow')
    plt.plot(timens,gs,label='G', alpha=lalpha); plt.plot(timens,es,label='E', alpha=lalpha)
    plt.plot(timens,hos,label='H', alpha=lalpha); plt.plot(timens,fts,label='F', alpha=lalpha)
    plt.grid()
    plt.xlim(tstart,12.5+tstart)
    plt.xlabel("Time (ns)"); plt.ylabel(r'Carrier Density (cm$^{-3}$)')
    plt.ylim(1e14,1e20)
    plt.yscale('log')
    
    plt.subplot(224)
    plt.title("Steady State")
    tstart = 231.1
    plt.axvspan(tstart,tstart+12.5,alpha=palpha, color = 'teal')
    plt.plot(timens,gs,label='G', alpha=lalpha); plt.plot(timens,es,label='E', alpha=lalpha)
    plt.plot(timens,hos,label='H', alpha=lalpha); plt.plot(timens,fts,label='F', alpha=lalpha)
    plt.grid()
    plt.xlim(tstart,12.5+tstart)
    plt.xlabel("Time (ns)"); plt.ylabel(r'Carrier Density (cm$^{-3}$)')
    plt.ylim(1e14,1e20)
    plt.yscale('log')
    
    plt.show()

#--- Show signal evolution per cycle ------------------------------------------
if 0: 
    cwrate = 0
    gs, es, hos, fts, gsigs, ehsigs = integration()
    plt.figure()
    for n in np.array([1,2,3,5,10,15,20,25,30]): 
        tstart = 6.1+12.5*(n-1)
        print(n)
        timens = time*1e9
        timens-= tstart
        plt.title("PL signal evolution")
        plt.plot(timens, (gsigs+ehsigs), alpha = 0.6, label = n)
    plt.grid()
    plt.xlim(-0.1,12.4)
    plt.ylabel(r'Outcoming photons cm$^{-3}$ s$^{-1}$ ($k_{hcat}GH + k_{rec}EH$)')
    plt.yscale('log')
    plt.legend(loc=4,title='Cycle')
    plt.xlabel('Time (ns)')
    plt.show()

#--- Show separate graphs for populations with variable CW power --------------
if 0:
    lalpha = 0.7
    tstart = 231.1
    timens -= tstart
    plt.figure()
    
    for cwrate in np.array([0,5e25,5e26,5e27,5e28]):
        print(cwrate)
        cwlabel = cwlabels(cwrate)
        
        gs, es, hos, fts, gsigs, ehsigs = integration()
        
        plt.subplot(221)
        plt.title("Geminate Pairs")
        plt.grid()
        plt.xlabel("Time (ns)"); plt.ylabel(r'Density (cm$^{-3}$)')
        plt.xlim(-0.1,12.4)
        plt.ylim(1.5e18,1.5e20)
        plt.yscale('log')
        plt.plot(timens, gs, label = cwlabel, alpha = lalpha)
        
        plt.subplot(222)
        plt.title("Electrons")
        plt.plot(timens, es, label = cwlabel, alpha = lalpha)
        plt.xlabel("Time (ns)"); plt.ylabel(r'Density (cm$^{-3}$)')
        plt.xlim(-0.1,12.4)
        plt.ylim(1.2e19,4e19)
        plt.yscale('log')
        plt.grid()
        
        plt.subplot(223)
        plt.title("Holes")
        plt.plot(timens, hos, label = cwlabel, alpha = lalpha)
        plt.xlabel("Time (ns)"); plt.ylabel(r'Density (cm$^{-3}$)')
        plt.xlim(-0.1,12.4)
        plt.ylim(1.5e19,3.7e19)
        plt.yscale('log')
        plt.grid()
        
        plt.subplot(224)
        plt.title("Filled Traps")
        plt.plot(timens, fts, label = cwlabel, alpha = lalpha)
        plt.xlabel("Time (ns)"); plt.ylabel(r'Density (cm$^{-3}$)')
        plt.xlim(-0.1,12.4)
        plt.ylim(5.15e17,5.25e17)
        plt.yscale('linear')
        plt.grid()
        
        plt.legend()
    plt.show()

#--- Show difference in PL signal for varying CW intensity --------------------
if 1:
    lalpha = 0.7
    tstart = 231.1
    timens -= tstart
    plt.figure()
    for cwrate in np.array([0,5e25,5e26,5e27,5e28]):
        print('cwrate ', cwrate)
        cwlabel = cwlabels(cwrate)
        
        gs, es, hos, fts, gsigs, ehsigs = integration()
        
        plt.plot(timens, gsigs+ehsigs, label = cwlabel,alpha = lalpha)
    plt.grid()
    plt.ylabel(r'Outcoming photons cm$^{-3}$ s$^{-1}$ ($k_{hcat}GH + k_{rec}EH$)')
    plt.xlabel('Time (ns)')
    plt.yscale('log')
    plt.xlim(-0.1,12.4)
    plt.ylim(2e26,1e29)
    plt.legend()
    plt.show()