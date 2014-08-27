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

#specify a Chabrier LF, but given in dN/dM. The Chabrier IMF is given typically as dN/d(logM)
#dN/dM = (1/ln10)*(1/M)*dN/dlogM, and this is calculated within the function. Finally, return
#dM, as for f_salpeter .
#Careful: denominator in first term has ln10 = np.log(10), but exponential is log10 M, so np.log10(m)
def f_chabrier(mass_arr,mass_min,mass_max,mass_crit,sigma_mass_crit):
    dmass_arr = np.ediff1d(mass_arr,to_end=0.0)  
    dmass_arr[len(dmass_arr)-1] = dmass_arr[len(dmass_arr)-2] 
    dN_arr = ((1./(np.log(10.)*mass_arr)) * (1./(np.sqrt(2.*np.pi)*sigma_mass_crit)) * 
        np.exp(-1. * (np.log10(mass_arr)-np.log10(mass_crit))**2 / (2. * sigma_mass_crit**2)) * 
        dmass_arr)
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

dN_arr1 = f_chabrier(mass_arr,0.1,0.8,0.2,0.2)
dN_arr2 = f_chabrier(mass_arr,0.1,0.8,0.2,0.4)
dN_arr3 = f_chabrier(mass_arr,0.1,0.8,0.6,0.2)

plt.semilogy(mass_arr,dN_arr1,label='mc=.2,sig=.2',color='red')
plt.semilogy(mass_arr,dN_arr2,label='mc=.2,sig=.4',color='blue')
plt.semilogy(mass_arr,dN_arr3,label='mc=.6,sig=.2',color='green')
plt.xlabel('Mass [Msun]') ; plt.ylabel('dN')
plt.title('Chabrier, Test Code')
plt.show()



