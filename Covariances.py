# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import PSim
import pickle
import csv
import simplex

fit_type = 'global'
scale = 0.003

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


def scalar_min(p, data):
    xdata, ydata, ysim = data[0]
    xdata_250, ydata_250, ysim_250 = data[1]
    scaled_ysim = ysim * p[0]
    scaled_ysim_250 = ysim_250 * p[0]
    err_20 = 0
    err_250 = 0
    num_points = 0
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
    for dat, sim in zip(ydata_250[:-1], scaled_ysim_250[:-1]):
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
    err = np.sqrt(err_250*err_20)
    if np.isnan(err):
        err = 7e20
    fitness = err * 100
    return fitness


def evaluate(p):
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
        sim.G3loss = p[9] * sim.step
        sim.scalar = 1
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
        sim_250.G3loss = p[9] * sim_250.step
        sim_250.scalar = 1
        for power in powers_250:
            sim_250.addPower(power)
        sim_250.runSim()
        interp_signals_250 = []
        for this_run in sim_250.signal:
            interp_this = np.interp(xarray_250, sim_250.xdata, this_run)
            interp_signals_250.append(interp_this)
        interp_signals_250 = np.array(interp_signals_250)
        data[1] = [xarray_250, yarrays_250, interp_signals_250]
    # Use a simplex minimization to find the best scalar
    scalar0 = np.array([3e-26])
    ranges = scalar0*0.1
    s = simplex.Simplex(scalar_min, scalar0, ranges)
    values, fitness, iter = s.minimize(epsilon=0.00001, maxiters=500,
                                       monitor=0, data=data)
    scalar = values[0]
    #p[-1] = scalar
    if scalar < 0:
        fitness = 1e30
    return fitness


def main():
    logname = 'best_{}.log'.format(fit_type)
    with open(logname, 'rb') as best_file:
        reader = csv.reader(best_file, dialect='excel-tab')
        p0 = []
        for val in reader.next():
            p0.append(np.float(val))
    dim = 11
    pi = np.ones(dim)
    for i, n in enumerate([0,1,2,3,4,5,6,7,8,9,10]):
        pi[i] = p0[n]
    ps1 = np.ndarray([dim, dim, dim])
    ps2 = np.ndarray([dim, dim, dim])
    fitness1 = np.ndarray([dim, dim])
    fitness2 = np.ndarray([dim, dim])
    differences = scale*pi
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                val1 = pi[k]
                val2 = pi[k]
                if i == k or j == k:
                    val1 = val1 + differences[k]
                    val2 = val2 - differences[k]
                ps1[i][j][k] = val1
                ps2[i][j][k] = val2
    for i in range(dim):
        for j in range(i, dim):
            fitness1[i][j] = evaluate(ps1[i][j])
            fitness1[j][i] = fitness1[i][j]
            fitness2[i][j] = evaluate(ps2[i][j])
            fitness2[j][i] = fitness2[i][j]
    error0 = evaluate(pi)
    data = {'fitness1': fitness1,
            'fitness2': fitness2,
            'differences': differences,
            'error0': error0}
    with open("covariance_data_{}.p".format(scale), "wb") as file:
        pickle.dump(data, file)
    hessian = np.ndarray([dim, dim])
    for i in range(dim):
        for j in range(dim):
            if i == j:
                d2i = differences[i]
                df1 = (fitness1[i][j] - error0) / d2i
                df2 = (error0 - fitness2[i][j]) / d2i
                hessian[i][j] = (df1 - df2) / (d2i)
            else:
                df1di1 = (fitness1[i][i] - error0) / differences[i]
                df1di2 = (fitness1[i][j] - fitness1[j][j]) / differences[i]
                dff1didj = (df1di2 - df1di1) / differences[j]
                df2di1 = (error0 - fitness2[i][i]) / differences[i]
                df2di2 = (fitness2[j][j] - fitness2[i][j]) / differences[i]
                dff2didj = (df2di2 - df2di1) / differences[j]
                hessian[i][j] = (dff1didj + dff2didj) / 2
                hessian[j][i] = hessian[i][j]
    with open("hessian_{}.p".format(scale), "wb") as file:
        pickle.dump(hessian, file)
    m_hessian = np.matrix(hessian)
    covariance = np.linalg.inv(m_hessian)
    cv_array = np.array(covariance)
    paramaters=['Traps', 'EH_Decay', 'E_Trap', 'TH_loss', 'G_Decay', 'G2_Decay', 'G3_Decay', 'GH_Decay', 'G_Escape', 'G3_Loss']
    for i in range(dim):
        print('{}{}: {} +- {}'.format(' ' * (8-len(paramaters[i])), paramaters[i], p0[i], np.sqrt(cv_array[i][i])))
    with open('Parameters_{}.txt'.format(scale), 'w') as f:
        writer = csv.writer(f, dialect="excel-tab")
        for i in range(10):
            error = np.sqrt(cv_array[i][i])
            relerror = error / pi[i] * 100
            words = '{}{}: {} +- {} ({}%)'.format(' ' * (8-len(paramaters[i])), paramaters[i], pi[i], error, relerror)
            print(words)
            writer.writerow([words])


if __name__ == '__main__':
    main()
