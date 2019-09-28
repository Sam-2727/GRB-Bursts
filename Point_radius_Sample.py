#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 17:43:45 2019

@author: samchristian
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from sklearn.metrics.pairwise import haversine_distances as haversine
import time
start_time = time.time()
data = pd.read_csv(r'/Users/samchristian/Downloads/grb-frompaper.csv')
#print(type(data))
data = data.loc[(data['z'] <= 2.1) & (data['z'] >= 1.6)]
zs1 = data.loc[(data['z'] <= 2.1) & (data['z'] >= 1.6)]
ras11 = data.loc[:, ['ra']]
decs11 = data.loc[:, ['dec']]
DFoutput = pd.concat([ras11, decs11], axis = 1)
#print(ras)
#print(decs)
#zs = zs1[[zs1 <= 2.1]]
#zs = zs[zs1 >= 1.6]
#ras = ras1[[zs1 <= 2.1]]
#ras = ras[[[zs1 >= 1.6]]]
#decs = decs1[[zs1 <= 2.1]]
#decs = decs[[zs1 >= 1.6]]
ras_inrange = []
decs_inrange = []
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
    #TODO: if this works make sure to change to radians in other 
    return c.ra.radian, c.dec.radian                                     #Many different formats are possible, e.g c.ra.hour for decimal hour values

def print_random_star_coords(nstars):
    #stars = []
    for n in range(nstars):
        RA , dec = random_point_on_sky()
        return [dec, RA]
i = 0
ras1 = []
decs1 = []
while i < len(ras11):
    #print(ras11.iloc[i].values[0])
    ras1.append(ras11.iloc[i]*np.pi/180)
    decs1.append(decs11.iloc[i]*np.pi/180)
    i += 1
test_radii = 10000
i = 0
radius = 0.8953539
test_points = print_random_star_coords(test_radii)
#print(test_points)
num_in_distance = []
max_coords = []
while i < test_radii:
    #print(i)
    in_distance = []
    test_point = print_random_star_coords(1)
    max_coords.append(test_point)
    #print(test_point)
    #test_point = test_points[i]
    j = 0
    #print(test_point)
    while j < len(ras1):
        #print(j)
        #print(ras1)
        distance = haversine([test_point, [decs1[j], ras1[j]]])
        #print(ras1.iloc[j], decs1.iloc[j])
        #print(distance)
        if distance[0][1] < radius:
            #print(distance[0][1])
            in_distance.append(1)
        j += 1
    num_in_distance.append(len(in_distance))
    i += 1
print(max_coords)
print(max(num_in_distance))
max_coord_index = num_in_distance.index(max(num_in_distance))
max_coord = max_coords[max_coord_index]
i = 0
coords_in_radius = []
while i < len(ras1):
    distance = haversine([max_coord, [decs1[i], ras1[i]]])
    if distance[0][1] < radius:
        coords_in_radius.append([ras1[i].values[0], decs1[i].values[0]])
    i += 1
print(coords_in_radius)
#279.95833*np.pi/180 273.90467*np.pi/180
#print(1)
#print(len(ras1))
outputradius = pd.DataFrame.from_records(coords_in_radius)        
print("--- %s seconds ---" % (time.time() - start_time))
outputradius.to_csv('/Users/samchristian/Downloads/grb-outputinradius.csv')
