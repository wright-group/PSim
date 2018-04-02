# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import PSim
import pickle
import csv
import simplex
import time


fit_type = 'global'

load_time = time.strftime('%Y.%m.%d.%H.%M')
filename = "simplex_{}_output_{}.log".format(fit_type, load_time)

with open("pickled_data.p", "r") as file:
    pickled_data = pickle.load(file)

powers = pickled_data['powers']
xdata = pickled_data['allxdata']
ydata = pickled_data['allydata']
xarray = pickled_data['xarray']
yarrays = pickled_data['yarrays']
averages = pickled_data['averages']
period = 50  # ns

with open("pickled_data_250.p", "r") as file:
    pickled_data_250 = pickle.load(file)

powers_250 = pickled_data_250['powers']
xdata_250 = pickled_data_250['allxdata']
ydata_250 = pickled_data_250['allydata']
xarray_250 = pickled_data_250['xarray']
yarrays_250 = pickled_data_250['yarrays']
averages_250 = pickled_data_250['averages']
period_250 = 1.0 / 250000.0 / 1e-9  # ns

ssn = 0


def scalar_min(data):
    xdata, ydata, scaled_ysim = data[0]
    xdata_250, ydata_250, scaled_ysim_250 = data[1]
    err_20 = 0
    err_250 = 0
    num_points = 0
    if (fit_type is 'global') or (fit_type is 20):  # 20 MHz data
        for dat, sim in zip(ydata, scaled_ysim):
            for x, d, s in zip(xdata, dat, sim):
                try:
                    if s > 0:
                        log_s = np.log(s)
                    else:
                        log_s = 0
                    log_d = np.log(d)
                    error = (log_s - log_d)
                    # error = np.log(error)
                    err_20 += error*error
                    num_points = num_points + 1
                except:
                    err_20 += 8e20
        err_20 = err_20 / num_points
        num_points = 0
    if (fit_type is 'global') or (fit_type is 250):  # 250 kHz data
        for dat, sim in zip(ydata_250[:-1], scaled_ysim_250[:-1]):  # Exclude the lowest noisy power
            for x, d, s in zip(xdata_250, dat, sim):
                try:
                    if s > 0:
                        log_s = np.log(s)
                    else:
                        log_s = 0
                    log_d = np.log(d)
                    error = (log_s - log_d)
                    # error = np.log(error)
                    if x >= -0.25 and x <= 120:
                        err_250 += error*error
                        num_points = num_points + 1
                except:
                    err_250 += 8e20
        err_250 = err_250 / num_points
    if fit_type is 'global':
        err = np.sqrt(err_250*err_20)
    elif fit_type is 20:
        err = err_20
    elif fit_type is 250:
        err = err_250
    else:
        err = 6e20
    if np.isnan(err):
        err = 7e20
    fitness = err * 100
    return fitness

def evaluate(p):
    global ssn
    dummy_x = np.zeros(10)
    dummy_y = np.zeros([10, 10])
    data = [[dummy_x, dummy_y, dummy_y], [dummy_x, dummy_y, dummy_y]]
    if fit_type is 'global' or fit_type is 20:  # 20 MHz data
        sim = PSim.DecaySim(reprate=20000000, tolerance=0.005, step=5e-12)
        sim.trap = p[0]
        sim.EHdecay = p[1] * sim.step
        sim.Etrap = p[2] * sim.step
        sim.FHloss = p[3] * sim.step
        sim.Gdecay = p[4] * sim.step
        sim.G2decay = p[5] * sim.step
        sim.G3decay = p[6] * sim.step
        sim.GHdecay = p[7] * sim.step
        sim.Gescape = p[8] * sim.step
        sim.Gform = p[9] * sim.step * 0
        sim.G3loss = p[10] * sim.step
        sim.scalar = p[11]
        for power in powers:
            sim.addPower(power)
        sim.runSim()
        interp_signals = []
        for this_run in sim.signal:
            interp_this = np.interp(xarray, sim.xdata, this_run)
            interp_signals.append(interp_this)
        interp_signals = np.array(interp_signals)
        data[0] = [xarray, yarrays, interp_signals]
    if fit_type is 'global' or fit_type is 250:  # 250 kHz data
        sim_250 = PSim.DecaySim(reprate=250000, tolerance=0.005, step=5e-12)
        sim_250.trap = p[0]
        sim_250.EHdecay = p[1] * sim_250.step
        sim_250.Etrap = p[2] * sim_250.step
        sim_250.FHloss = p[3] * sim_250.step
        sim_250.Gdecay = p[4] * sim_250.step
        sim_250.G2decay = p[5] * sim_250.step
        sim_250.G3decay = p[6] * sim_250.step
        sim_250.GHdecay = p[7] * sim_250.step
        sim_250.Gescape = p[8] * sim_250.step
        sim_250.Gform = p[9] * sim_250.step * 0
        sim_250.G3loss = p[10] * sim_250.step
        sim_250.scalar = p[11]
        for power in powers_250:
            sim_250.addPower(power)
        sim_250.runSim()
        interp_signals_250 = []
        for this_run in sim_250.signal:
            interp_this = np.interp(xarray_250, sim_250.xdata, this_run)
            interp_signals_250.append(interp_this)
        interp_signals_250 = np.array(interp_signals_250)
        data[1] = [xarray_250, yarrays_250, interp_signals_250]
    fitness = scalar_min(data=data)
    for param in p:
        if param < 0:
            fitness = fitness * 100000
    with open(filename, 'a') as log_file:
        writer = csv.writer(log_file, dialect="excel-tab")
        row = [ssn, '{:.4e}'.format(fitness)]
        for var in p:
            row.append('{:.4e}'.format(var))
        writer.writerow(row)
    ssn = ssn + 1
    return fitness

def main():
    try:
        logname = 'best_{}.log'.format(fit_type)
        with open(logname, 'rb') as best_file:
            reader = csv.reader(best_file, dialect='excel-tab')
            p0 = []
            for val in reader.next():
                p0.append(np.float(val))
        p0 = np.array(p0)
        ranges = p0*0.02
        s = simplex.Simplex(evaluate, p0, ranges)
        p, err, iter = s.minimize(epsilon=0.00001, maxiters=2000,
                                  monitor=0)
        with open(logname, 'a') as save_file:
            writer = csv.writer(save_file, dialect='excel-tab')
            writer.writerow(p)
            writer.writerow([ssn, err])
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()
