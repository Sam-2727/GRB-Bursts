#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 18:47:18 2019

@author: samchristian
GRB_nearestneighbor with random data
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 13:29:25 2019

@author: samchristian
KS test
"""
#TODO: For safety, do what horvath did and compare with other redshift ranges
import numpy as np
from numpy.random import power
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from sklearn.metrics.pairwise import haversine_distances as haversine
import pandas as pd
import time
import seaborn as sns
from scipy import stats
from statistics import mean
sns.set_color_codes()
data = pd.read_csv(r'/Users/samchristian/Downloads/grb-frompaper.csv')
#print(type(data))
data = data.loc[(data['z'] <= 2.1) & (data['z'] >= 1.6)]
zs1 = data.loc[(data['z'] <= 2.1) & (data['z'] >= 1.6)]
data_different = data.loc[(data['z'] <= 9.4) & (data['z'] >= 2.68)]
ras2 = data_different.loc[:, ['ra']].values
decs2 = data_different.loc[:, ['dec']].values
ras1 = data.loc[:, ['ra']].values
decs1 = data.loc[:, ['dec']].values
print(len(zs1))
def random_point_on_unit_sphere():
    while True:
        R   = np.random.rand(3) #Random point in box
        R   = 2*R - 1
        rsq = sum(R**2)
        if rsq < 1: break       #Use r only if |r|<1 in order not to favor corners of box
    return R / np.sqrt(rsq)     #Normalize to unit vector

def random_point_on_sky():
    p     = random_point_on_unit_sphere()
    r     = np.linalg.norm(p)
    theta = 90 - (np.arccos(p[2] / r)    / np.pi * 180)            #theta and phi values in degrees
    phi   =       np.arctan(p[1] / p[0]) / np.pi * 180
    c     = SkyCoord(ra=phi, dec=theta, unit=(u.degree, u.degree)) #Create coordinate
    return c.ra.deg, c.dec.deg                                     

def print_random_star_coords(nstars):
    for n in range(nstars):
        RA,dec = random_point_on_sky()
        return [dec, RA]
def nearest_neighbor(points):
    i = 0
    while i < len(points):
        distances = []
        point = points[i]
        #print(point)
        j = 0
        while j < len(points):
            if i == j:
                j += 1
                continue
            #print(points[j])
            #print(point, points[j])
            distance = haversine([point, points[j]])
            distances.append(distance[0][1])
            j += 1
        #print(distances)
        ascending = sorted(distances)
        for m in ascending:
            final_distributions.append(m)
        closest_points.append(ascending[22]) #change 29 to kth nearest neighbor
        i += 1

final_distributions = []
n = 0
#while n < 44:
#    final_distributions.append([])

        
normalize = 0.00276920001229
random = np.random.rand(100)
number1 = 3.85
number2 = (-1.07)
leading_term = 371.899529439
values = 3.33*power(3.85-1, 10000) - 1
values2 = 10*power(0.01, 100000) - 1
values3 = []
values4=[]
for b in values:
    if b > 0:
        values4.append(b)
for a in values2:
    if a > 1.33:
        values3.append(a)
#nobs = 10
i = 0
#final_values = []
sky_dist = []
sky_cords = []
print(i)
nobs = 44
while i < nobs:
    number = np.random.rand(1)
    if number < 0.804805705672:
        #index = np.random.choice(values3.shape, n, replace=False)
        sky_dist.append(np.random.choice(values3))
    if number > 0.804805705672:
        #index = np.random.choice(values.shape, n, replace=False)
        sky_dist.append(np.random.choice(values4))
    sky_cords.append(print_random_star_coords(1))    
    #print(print_random_star_coords(1))
    i += 1
test_radii = 10000
l = 0
radius = 0.8953539
#test_points = print_random_star_coords(test_radii)
#print(test_points)
points1 = []
num = 44
iter_2 = 100
while l < (num*iter_2 + 1): 
   # print(l)
    point = print_random_star_coords(1)
    #print(point)
    points1.append(point)
    l += 1
slice_number = 0
k = 0
points_iters = []
while k < iter_2:
    closest_points = []
    print(k)
    sample = points1[slice_number:(slice_number+num)]
    slice_number += num
    nearest_neighbor(sample) 
    points_iters.append(closest_points)
    k += 1
closest_points = []
density_function_actual = []
points_actual = []
for (i, j) in zip(decs1, ras1):
    #print(i)
    points_actual.append([i[0], j[0]])
#print(points_actual)
nearest_neighbor(points_actual)
density_function_actual.append(closest_points)
#print(closest_points)
n_bins = 50
fig, ax = plt.subplots(figsize=(8, 8))
#n, bins, patches = ax.hist(density_function_actual, n_bins, density=True, histtype='step',
#                           cumulative=True, label='Empirical')
statistics_k_value = []
statistics_k_value1 = []
#print(density_function_actual)
i = 0
while i < iter_2:
    compare_1 = points_iters[i]
    compare_2 = points_iters[i+1]
    statistics_k_value.append(stats.ks_2samp(compare_1, compare_2)[0])
    statistics_k_value1.append(stats.ks_2samp(compare_1, density_function_actual[0])[0])
    i += 2
#for num, i in enumerate(points_iters):
#    n, bins, patches = ax.hist(i, n_bins, density=True, histtype='step',
#                           cumulative=True, label='Empirical', color='green')
#    statistics_k_value.append(stats.ks_2samp(i, density_function_actual[0])[0])   
#plt.show()
#ax.hist(statistics_k_value, 20)
#ax.hist(statistics_k_value1, 20)
distribution = pd.Series(statistics_k_value1)
lessthan = distribution[distribution < mean(statistics_k_value)]
greaterthan = distribution[distribution > mean(statistics_k_value)]
for i in points_iters:
    sns.distplot(i, hist = False, hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True), bins = 50, color='g')
sns.distplot(density_function_actual, hist = False, hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True), bins = 50, color='r')
print(len(lessthan))
print(len(greaterthan))
print(len(distribution))
print(mean(statistics_k_value))
print(statistics_k_value)
print(statistics_k_value1)
