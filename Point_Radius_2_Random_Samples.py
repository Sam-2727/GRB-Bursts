#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 16:16:52 2019

@author: samchristian
GRB-sim
"""
import numpy as np
from numpy.random import power
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from sklearn.metrics.pairwise import haversine_distances as haversine
import pandas as pd
import time
start_time = time.time()
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
    return c.ra.radian, c.dec.radian                                   

def print_random_star_coords(nstars):
    for n in range(nstars):
        RA,dec = random_point_on_sky()
        return [dec, RA]
def bootstrap_point_method(points):
        i = 0
        num_in_distance = []
        while i < test_radii:
            #print(i)
            in_distance = []
            test_point = print_random_star_coords(1)
            #print(test_point)
            #test_point = test_points[i]
            j = 0
            while j < len(points):
                #print(len(points))
                #print(ras1)
                distance = haversine([test_point, points[j]])
                #print(ras1.iloc[j], decs1.iloc[j])
                #print(distance)
                if distance[0][1] < radius:
                    #print(i)
                    in_distance.append(1)
                j += 1
            num_in_distance.append(len(in_distance))
            #print(in_distance)
            i += 1
        maxes.append(max(num_in_distance))
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
nobs = 1000
i = 0
#final_values = []
sky_dist = []
sky_cords = []
print(i)
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
iter_2 = 1000
while l < (num*iter_2 + 1): 
   # print(l)
    point = print_random_star_coords(1)
    #print(point)
    points1.append(point)
    l += 1
#print(print_random_star_coords(1))
slice_number = 0
k = 0
maxes = []
print('i')
while k < iter_2:
    print(k)
    sample = points1[slice_number:(slice_number+num)]
    slice_number += num
    bootstrap_point_method(sample)
    k += 1
#print(sky_dist)
print(maxes)
plt.hist(sky_dist, 50)
print()
print("--- %s seconds ---" % (time.time() - start_time))
max_df = pd.DataFrame(maxes)

max_df.to_csv(r'/Users/samchristian/Downloads/GRB_results_bootstrap1002.csv')