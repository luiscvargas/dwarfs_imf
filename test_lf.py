import sys
import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import interpolate
import pandas as pd
import timeit

def read_iso():
    f = open('iso/test.iso','r')
    ids = ['mass','teff',';logg','g','r','i']
    iso_pd = pd.read_csv(f,comment='#',names=ids,delim_whitespace=True,
        header=0,skiprows=8,usecols=(1,2,3,6,7,8))
    iso_pd['gr'] = iso_pd['g'] - iso_pd['r']
    f.close()
    return iso_pd

iso = read_iso()

mass_min = 0.1 ; mass_max = 0.8

#fit a spline to M - r relation

tck = interpolate.splrep(iso[iso['mass'] < mass_max]['mass'], iso[iso['mass'] < mass_max]['r'])
mass_new = np.arange(mass_min,mass_max,0.01)
r_new = interpolate.splev(mass_new,tck,der=0)
dr_new = interpolate.splev(mass_new,tck,der=1)

#plot original data and spline

plt.plot(iso['mass'],iso['r'],color='b',markersize=5,marker='^')
plt.plot(mass_new,r_new,color='r',markersize=3,marker='o')
plt.plot(mass_new,dr_new,color='g',markersize=3,marker='o')
plt.show()


