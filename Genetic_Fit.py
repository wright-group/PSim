# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 14:12:46 2015

@author: matt
"""
from __future__ import division
import multiprocessing
import multiprocessing.pool
import numpy as np
import PSim
import pickle
import csv
import simplex
import time

fit_type = 'global'


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

load_time = time.strftime('%Y.%m.%d.%H.%M')
filename = "genetic_{}_output_{}.log".format(fit_type, load_time)

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

SSN = 0  # Unique ID for each individual in a population


class Individual():

    def __init__(self, p0, clone=False):
        global powers
        global SSN
        self.ssn = SSN
        SSN = SSN + 1
        self.p = []
        if not clone:
            for param in p0:
                exponent = np.random.normal(0, 0.20)
                new_paramater = param * np.exp(exponent)
                self.p.append(new_paramater)
        else:
            self.p = p0
        self.p[-1] = 1
        self.fitness = 0

    def setFitness(self, fitness):
        self.fitness = fitness

    def setScalar(self, scalar):
        self.p[-1] = scalar

    def clone_self(self):
        clone = Individual(self.p, clone=True)
        return clone

    def mutate(self):
        mutated = False
        for i, param in enumerate(self.p[:-1]):
            if np.random.rand() < 0.15:
                self.p[i] = np.random.normal(param, param * 0.01)
                mutated = True
        return mutated

    def crossover(self, other):
        child1 = self.clone_self()
        child2 = other.clone_self()
        number_exchanged = 2
        changed = False
        for i in range(number_exchanged):
            param_num = np.random.randint(len(self.p)-1)
            if not child1.p[param_num] == child2.p[param_num]:
                child1_param = child1.p[param_num]
                child1.p[param_num] = child2.p[param_num]
                child2.p[param_num] = child1_param
                changed = True
        return child1, child2, changed


def evaluate(p, ssn):
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
        sim.Gform = p[9] * sim.step
        sim.G3loss = p[10] * sim.step
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
        sim_250.Gform = p[9] * sim_250.step
        sim_250.G3loss = p[10] * sim_250.step
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
    scalar0 = np.array([8e-21])
    ranges = scalar0*0.1
    s = simplex.Simplex(scalar_min, scalar0, ranges)
    values, fitness, iter = s.minimize(epsilon=0.00001, maxiters=500,
                                       monitor=0, data=data)
    scalar = values[0]
    p[-1] = scalar
    if scalar < 0:
        fitness = 1e30
    with open(filename, 'a') as log_file:
        writer = csv.writer(log_file, dialect="excel-tab")
        row = [ssn, '{:.4e}'.format(fitness)]
        for var in p:
            row.append('{:.4e}'.format(var))
        writer.writerow(row)
    return fitness, scalar


def minimize(p0, pop_size=2, generations=2, processes=4):
    crossover_probability = 0.02
    mutation_probability = 0.1
    new_probability = 0.75
    pop = []
    pop.append(Individual(p0, clone=True))
    for i in range(pop_size):
        pop.append(Individual(p0))

    # Evaluate the entire population
    pool = MyPool(processes=processes)
    jobs = []
    for individual in pop:
        inputs = (individual.p, individual.ssn)
        jobs.append(pool.apply_async(evaluate, args=inputs))
    pool.close()
    pool.join()

    for job, individual in zip(jobs, pop):
        fitness, scalar = job.get()
        individual.setFitness(fitness)
        individual.setScalar(scalar)

    for gen in range(generations):
        offspring = []
        # create a new population member and mix it with the best
        if np.random.rand() < new_probability:
            new_member = Individual(pop[0].p)
            offspring.append(new_member)
            child1, child2, changed = new_member.crossover(pop[0])
            if changed:
                offspring.append(child1)
                offspring.append(child2)

        # iterate over each individual in the population
        for i, individual in enumerate(pop):
            if np.random.rand() < crossover_probability:
                # Crossover with a random, non-identical partner
                partner = np.random.randint(len(pop))
                while partner == i:
                    partner = np.random.randint(len(pop))
                child1, child2, changed = individual.crossover(pop[partner])
                if changed:
                    offspring.append(child1)
                    offspring.append(child2)
            if np.random.rand() < mutation_probability:
                # Create a mutant
                mutant = individual.clone_self()
                if mutant.mutate():
                    offspring.append(mutant)

        # Evaluate the offspring
        if len(offspring) > 0:
            pool = MyPool(processes=processes)
            jobs = []
            for individual in offspring:
                inputs = (individual.p, individual.ssn)
                jobs.append(pool.apply_async(evaluate, inputs))
            pool.close()
            pool.join()
            for job, individual in zip(jobs, offspring):
                fitness, scalar = job.get()
                individual.setFitness(fitness)
                individual.setScalar(scalar)
        new_pop = pop + offspring

        def find_value(ind):
            return ind.fitness
        new_pop.sort(key=find_value)
        '''
        # Ensure Genetic Diversity! - Because they are already sorted, we only
        #                             need to compare neighbors
        previous_p = p0[:-1] * 0.00 # Exclude the scalar
        for individual in new_pop:
            this_p = individual.p[:-1]  # Exclude the scalar
            if (this_p == previous_p).all():
                new_pop.remove(individual)
            else:
                previous_p ==  this_p
        '''
        pop = new_pop[:pop_size]
    return pop


def main():
    try:
        logname = 'best_{}.log'.format(fit_type)
        with open(logname, 'rb') as best_file:
            reader = csv.reader(best_file, dialect='excel-tab')
            p0 = []
            for val in reader.next():
                p0.append(np.float(val))
        p0 = np.array(p0)
        pop = minimize(p0, pop_size=500, generations=200, processes=4)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()
