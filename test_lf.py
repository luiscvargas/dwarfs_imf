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

#now specify a Salpeter LF, alpha is exponent in linear eqn, alpha = Gamma + 1

def f_salpeter(mass_arr,mass_min,mass_max,alpha):
    dmass_arr = np.ediff1d(mass_arr,to_end=0.0)  #to end sets last element to 0 otherwise
       #one element too few.
    dmass_arr[len(dmass_arr)-1] = dmass_arr[len(dmass_arr)-2] #update last element
    dN_arr = (mass_arr**(-1.*alpha)) * dmass_arr
    dN_arr[mass_arr < mass_min] = 0.0
    dN_arr[mass_arr > mass_max] = 0.0
    return dN_arr

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

#now specify a Salpeter LF

mass_arr = np.arange(0.0,2.0,0.001)
dN_arr1 = f_salpeter(mass_arr,0.15,1.5,2.35)
dN_arr2 = f_salpeter(mass_arr,0.15,1.5,2.00)
dN_arr3 = f_salpeter(mass_arr,0.15,1.5,2.60)

plt.semilogy(mass_arr,dN_arr1,label='alpha=2.35',color='red')
plt.semilogy(mass_arr,dN_arr2,label='alpha=2.00',color='blue')
plt.semilogy(mass_arr,dN_arr3,label='alpha=2.60',color='green')
plt.xlabel('Mass [Msun]') ; plt.ylabel('dN')
plt.title('Salpeter, Test Code')
plt.show()




