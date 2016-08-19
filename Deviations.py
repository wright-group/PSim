# -*- coding: utf-8 -*-
from __future__ import division
import multiprocessing
import multiprocessing.pool
import numpy as np
import PSim
import pickle
import csv
import simplex
from matplotlib import pyplot as plt

fit_type = 'global'
logname = 'best_{}.log'.format(fit_type)
with open(logname, 'rb') as best_file:
    reader = csv.reader(best_file, dialect='excel-tab')
    p0 = []
    for val in reader.next():
        p0.append(np.float(val))
p0 = np.array(p0[:-1])
dim = 10
pi = np.ones(dim)
for i, n in enumerate([0,1,2,3,4,5,6,7,8,10]):
    pi[i] = p0[n]
differences = 0.001*pi
with open("pickled_data_{}.p".format(fit_type), "r") as file:
    pickled_data = pickle.load(file)
with open("covariance_data_0.001.p".format(fit_type), "r") as file:
    pickled_data = pickle.load(file)
fitness1 = pickled_data['fitness1']
fitness2 = pickled_data['fitness2']
error0 = pickled_data['error0']
hessian = np.ndarray([10, 10])
for i in range(10):
    for j in range(10):
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
with open("pickled_hessian_{}.p".format(fit_type), "wb") as file:
    pickle.dump(hessian, file)
m_hessian = np.matrix(hessian)
covariance = np.linalg.inv(m_hessian)
cv_array = np.array(covariance)
paramaters=['Traps', 'EH_Decay', 'E_Trap', 'TH_loss', 'G_Decay', 'G2_Decay', 'G3_Decay', 'GH_Decay', 'G_Escape', 'G_Form', 'G3_Loss']
with open('Parameters.txt', 'w') as f:
    writer = csv.writer(f, dialect="excel-tab")
    for i in range(10):
        error = np.sqrt(cv_array[i][i])
        relerror = error / p0[i] * 100
        words = '{}{}: {} +- {} ({}%)'.format(' ' * (8-len(paramaters[i])), paramaters[i], p0[i], error, relerror)
        print(words)
        writer.writerow([words])

#%%
errors = np.zeros_like(fitness1)
for i in range(10):
    for j in range(10):
        e1 = fitness1[i][j]-error0
        e2 = error0-fitness2[i][j]
        diff = (e1-e2)**2
        errors[i][j]=diff
print("max_diff:{}".format(np.max(errors)))
print(errors[3][3])
print(fitness1[3][3]-error0)
plt.figure()
new_array = np.ones_like(cv_array)
for i in range(10):
    for j in range(10):
        if not i == j:
            new_array[i][j] = cv_array[i][j] / p0[i] / p0[j]
        else:
            new_array[i][j] = cv_array[i][j] / p0[i] / p0[j]
new_array = np.sqrt(np.abs(new_array))
new_array2 = np.delete(new_array, [4,6], axis=0)
new_array2 = np.delete(new_array2, [4,6], axis=1)
ave = np.average(new_array2)
for i in range(10):
    for j in range(10):
        if i == 4 or i == 6 or j == 4 or j == 6 or i==j:
            new_array[i][j] = ave
levels = np.linspace(np.min(new_array), np.max(new_array), 50)
plt.contourf(new_array, levels=levels)
plt.colorbar()
plt.grid(which='major')
plt.title("Normalized Covariance")
plt.show()
