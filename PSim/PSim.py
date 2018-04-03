# -*- coding: utf-8 -*-
"""
PSim.py - The Python Semiconductor Dynamics Simulator
Created on Mon Mar 09 10:07:36 2015

@author: Matthew Rowley
"""


# --- import -------------------------------------------------------------------------------------


import numpy as np

from multiprocessing.pool import ThreadPool as Pool


# --- define -------------------------------------------------------------------------------------


__all__ = ['DecaySim']


# --- workspace ----------------------------------------------------------------------------------


# Some universal constants
ps = 10.0**-12
ns = 10.0**-9
sqrt_2pi = np.sqrt(2 * np.pi)


def bEqn(g, e, h, Gdecay, G2decay, G3decay, GHdecay, Gescape, G3loss, Gform,
         pulse, step, stepsize):
    '''
    High concentration gem_pair master rate equation
    '''
    Gchange = (pulse[step]
               - g * Gdecay
               - g * g * G2decay
               - g * g * g * G3decay
               - g * h * GHdecay
               - g * Gescape
               - g * g * g *G3loss
               + e * h * Gform)
    return Gchange * stepsize


def hEqn(g, e, h, f, Gescape, Gform, EHdecay, FHloss, stepsize):
    """
    Hole master rate equation
    """
    hchange = (g * Gescape
               - h * e * EHdecay
               - h * f * FHloss
               - e * h * Gform)
    return hchange * stepsize


def eEqn(g, e, h, f, trap, Gescape, Gform, EHdecay, Etrap, stepsize):
    """
    Electron master rate equation
    """
    echange = (g * Gescape
               - h * e * EHdecay
               - e * (trap - f) * Etrap
               - e * h * Gform)
    return echange * stepsize


def fEqn(e, h, f, trap, Etrap, FHloss, stepsize):
    """
    Filled trap master rate equation
    """
    fchange = (e * (trap - f) * Etrap
               - f * h * FHloss)
    return fchange * stepsize


def gsignalEqn(g, h, Gdecay, G2decay, G3decay, GHdecay):
    signal = (g * Gdecay
              + g * g * G2decay
              + g * g * g * G3decay
              + g * h * GHdecay)
    return signal


def ehsignalEqn(e, h, EHdecay):
    """
    E-H Recombination Signal
    """
    signal = (h * e * EHdecay)
    return signal # Note that the scalar is not included here


def glossEqn(g, G3loss):
    gloss = g * g * g * G3loss
    return gloss


def tlossEqn(f, h, FHloss):
    '''
    Filled trap loss
    '''
    tloss = f * h * FHloss
    return tloss


def steadyYet(newg, oldg, newe, olde, newh, oldh, newf, oldf, tolerance):
    """
    Test if the simulation has reached steady state yet.
    """
    steady_yet = True
    if oldg == 0 or (abs(newg-oldg)/oldg > tolerance or
                     abs(newe-olde)/olde > tolerance or
                     abs(newh-oldh)/oldh > tolerance or
                     abs(newf-oldf)/oldf > tolerance):
        steady_yet = False
    return steady_yet


def qOverk(g, e, h, Keq):
    q = e*h/g
    return q/Keq


def powerRun(power, pulse, steps, trap, tolerance, EHdecay, Etrap, FHloss,
             Gdecay, G2decay, G3decay, GHdecay, Gescape, Gform, G3loss, Keq,
             trackQ, verbose):
    numsteps = len(steps)
    signal = np.zeros(numsteps)
    gsignal = np.zeros(numsteps)
    ehsignal = np.zeros(numsteps)
    gloss =  np.zeros(numsteps)
    tloss = np.zeros(numsteps)
    gem_pair = np.zeros(numsteps)
    electron = np.zeros(numsteps)
    hole = np.zeros(numsteps)
    filled = np.zeros(numsteps)
    qk = np.zeros(numsteps)
    failed = False
    new_g, new_e, new_h, new_f = 1, 1, 1, 1
    old_g, old_e, old_h, old_f = 1, 1, 1, 1
    runs = 0
    steady = False
    while not steady:
        runs = runs + 1
        for s, stepsize in enumerate(steps):  # s for step
            # Calculate the signal for this time step
            try:
                gsignal[s] = gsignalEqn(new_g, new_h, Gdecay, G2decay,
                                        G3decay, GHdecay)
                ehsignal[s] = ehsignalEqn(new_e, new_h, EHdecay)
                gloss[s] = glossEqn(new_g, G3loss)
                tloss[s] = tlossEqn(new_f, new_h, FHloss)
                signal[s] = gsignal[s] + ehsignal[s]
                if trackQ:
                    qk[s] = qOverk(new_g, new_e, new_h, Keq)
                # calculate changes in populations for the next time step
                hchange = hEqn(new_g, new_e, new_h, new_f, Gescape, Gform,
                               EHdecay, FHloss, stepsize)
                echange = eEqn(new_g, new_e, new_h, new_f, trap, Gescape,
                               Gform, EHdecay, Etrap, stepsize)
                fchange = fEqn(new_e, new_h, new_f, trap, Etrap, FHloss,
                               stepsize)
                gchange = bEqn(new_g, new_e, new_h, Gdecay, G2decay,
                               G3decay, GHdecay, Gescape, G3loss, Gform,
                               pulse, s, stepsize)
            except Exception as e:
                print("Failed by {}".format(e))
                failed = True
                break

            new_f = filled[s] + fchange
            new_h = hole[s] + hchange
            new_e = electron[s] + echange
            new_g = gem_pair[s] + gchange
            # Update the values for the next time step
            if s + 1 < numsteps:
                gem_pair[s+1] = new_g
                hole[s+1] = new_h
                electron[s+1] = new_e
                filled[s+1] = new_f
            else:  # Set up initial state for next round.
                old_g = gem_pair[0]
                gem_pair[0] = new_g
                old_h = hole[0]
                hole[0] = new_h
                old_e = electron[0]
                electron[0] = new_e
                old_f = filled[0]
                filled[0] = new_f
        steady = steadyYet(new_g, old_g, new_e, old_e, new_h,
                           old_h, new_f, old_f, tolerance)
    if verbose:
        print("Runs to steady state: {}".format(runs))
    return gem_pair, electron, hole, filled, signal, gsignal, ehsignal, gloss, tloss, qk


class DecaySim():
    """
    This class creates a simulation of semiconductor excited state transients.
    Solving the master rate equations numerically allows for quickly exploring
    the consequences of different decay rates or master rate equations.
    """
    def __init__(self, trap=2.5*10**16, Keq=1.0*10**17,
                 EHdecay=1.0*10**-10, Etrap=2.0*10**-10, FHloss=8.0*10**-12,
                 G3decay = 0, step=200*ps, pretime=2, reprate=80000000,
                 verbose=False, trackQ=False, scalar=1, Gdecay=0, GHdecay=0,
                 tolerance=0.005, G2decay=0. ,Gescape=1., Gform=1., G3loss=0.):
        """
        Initialization script sets up the parameters
        """
        # Some other variables used
        self.tolerance = tolerance
        self.scalar = scalar
        self.verbose = verbose
        self.reprate = reprate
        self.duration = 1.00 / reprate
        self.step = step
        self.steps = int(self.duration / self.step)
        self.powers = []
        self.pretime = pretime
        # Variables which hold state densities
        self.exciton = []
        self.hole = []
        self.electron = []
        self.trap = (trap)  # Total number of traps
        self.filled = []  # Filled traps
        self.signal = []
        self.xsignal = []
        self.ehsignal = []
        self.xloss = []
        self.tloss = []
        self.pulses = []
        self.qk = []
        self.trackQ = trackQ
        # Rate and equilibrium constants, corrected for time step size
        self.Keq = Gescape/Gform  # Equilibrium constant for X<-->e+h
        self.EHdecay = (EHdecay * step)  # e+h->ground
        self.Etrap = (Etrap * step)  # e+trap->filled
        self.FHloss = (FHloss * step)  # filled+h->ground
        self.Gdecay = Gdecay * step
        self.G2decay = G2decay * step
        self.G3decay = G3decay * step
        self.GHdecay = GHdecay * step
        self.Gescape = Gescape * step
        self.G3loss = G3loss * step
        self.Gform = Gform * step

    def runSim(self):
        """
        Generate the data arrays for all powers
        """
        if self.verbose:
            print("Running Simulation, This may take a while")
        self.makeXData(float(self.pretime))
        pool = Pool(processes=len(self.powers))
        jobs = []
        self.gem_pair = []
        self.electron = []
        self.hole = []
        self.filled = []
        self.signal = []
        self.gsignal = []
        self.ehsignal = []
        self.gloss = []
        self.tloss = []
        self.qk = []
        for power, pulse in zip(self.powers, self.pulses):
            inputs = [power, pulse, self.steps, self.trap, self.tolerance,
                      self.EHdecay, self.Etrap, self.FHloss, self.Gdecay,
                      self.G2decay, self.G3decay, self.GHdecay, self.Gescape,
                      self.Gform, self.G3loss, self.Keq, self.trackQ,
                      self.verbose]
            jobs.append(pool.apply_async(powerRun, inputs))
        for job in jobs:
            gem_pair, electron, hole, filled, signal, gsignal, ehsignal, gloss, tloss, qk = job.get()
            self.signal.append(signal * self.scalar / self.step)
            self.gsignal.append(gsignal * self.scalar / self.step)
            self.ehsignal.append(ehsignal * self.scalar / self.step)
            self.gloss.append(gloss * self.scalar / self.step)
            self.tloss.append(tloss * self.scalar / self.step)
            self.gem_pair.append(gem_pair)
            self.electron.append(electron)
            self.hole.append(hole)
            self.filled.append(filled)
            self.qk.append(qk)
        pool.close()

    def makeXData(self, pretime):
        """
        Generate the xdata (time) array, which is shared by all powers in
        the sim
        """
        xdata = []
        steps = []
        self.pulses = []
        time = - pretime * ns  # start the time
        while time < 2 * ns:  # around the pulse, just take minimum size steps
            xdata.append(time / ns)
            steps.append(1)
            time = time + self.step
        while time < self.duration - self.pretime * ns:
            xdata.append(time / ns)
            stepsize = int(np.log(time/self.step)*2)
            steps.append(stepsize)
            time = time + stepsize*self.step
        self.xdata = np.array(xdata)
        self.steps = np.array(steps)
        for power in self.powers:
            this_pulse = np.zeros_like(self.xdata)
            for i, time in enumerate(self.xdata):
                this_pulse[i] = self.excitationPulse(time, power)
            self.pulses.append(this_pulse)


    def addPower(self, power):
        """
        Add a power to simulate. The value should be the density of carriers
        generated by the pulse, in units of  N/cm^3
        """
        self.powers.append(float(power))

    def resetSim(self):
        """
        Remove all the added powers by reinitializing the powers list
        """
        self.powers = []

    def excitationPulse(self, time, power):
        """
        A function describing the excitation pulse.
        """
        t = time * ns + self.step  # Should center at one step before 0
        if self.step <= 200 * ps:  # resolution warrants modelling the pulse
            width = 200.0 * ps  # self.step

            if t < width * 10:  # Only evaulate when the value is significant
                amp = power / (width * sqrt_2pi)  # normalized amplitude
                value = amp * np.exp(-1.0 * (t) * (t) / (2 * width * width))
                value = value
            else:
                value = 0.0
        else:  # impulsive limit, just dump all the excitons in at t=0
            # if time >= 0 - self.step/2 and time < 0 + self.step/2:
            if t > -0.5 * self.step and t <= 0.5 * self.step:
                value = power / self.step
            else:
                value = 0.0
        return (value*self.step)
