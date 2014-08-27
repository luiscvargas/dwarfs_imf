import sys
import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import interpolate
import pandas as pd
import timeit

def read_iso():
    f = open('iso/darth_1gy.iso','r')
    ids = ['mass','teff',';logg','g','r','i']
    iso_pd = pd.read_csv(f,comment='#',names=ids,delim_whitespace=True,
        header=None,skiprows=9,usecols=(1,2,3,6,7,8))
    iso_pd['gr'] = iso_pd['g'] - iso_pd['r']
    f.close()
    return iso_pd

def read_lf():
    f = open('iso/darth_1gy.lf','r')
    ids = ['r','logN','logdN']
    lf_pd = pd.read_csv(f,comment='#',names=ids,delim_whitespace=True,
        header=None,skiprows=9,usecols=(1,2,3))
    f.close()
    return lf_pd

#now specify a Salpeter LF, alpha is exponent in linear eqn, alpha = Gamma + 1

def f_salpeter(mass_arr,mass_min,mass_max,alpha):
    dmass_arr = np.ediff1d(mass_arr,to_end=0.0)  #to end sets last element to 0 otherwise
       #one element too few.
    dmass_arr[len(dmass_arr)-1] = dmass_arr[len(dmass_arr)-2] #update last element
    dmass_arr = abs(dmass_arr)
    plt.plot(mass_arr,dmass_arr)
    plt.xlim(0,1.6)
    plt.ylim(0,0.2)
    plt.show()
    dN_arr = (mass_arr**(-1.*alpha)) * dmass_arr
    dN_arr[(mass_arr < mass_min) & (mass_arr > mass_max)] = 0.0
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
lf  = read_lf()

mass_min = 0.05 ; mass_max = 1.4
r_min_index = np.argmin(np.abs(iso['mass']-mass_max))
r_max_index = np.argmin(np.abs(iso['mass']-mass_min))
r_min = iso['r'][r_min_index]
r_max = iso['r'][r_max_index]

print r_min_index,r_max_index
print r_min,r_max

#fit a spline to M - r relation

tck = interpolate.splrep(iso[iso['mass'] < mass_max]['mass'], iso[iso['mass'] < mass_max]['r'])
mass_new = np.arange(mass_min,mass_max,0.01)
r_new = interpolate.splev(mass_new,tck,der=0)
dr_new = interpolate.splev(mass_new,tck,der=1)

#fit a spline to r - M relation

in_r = iso[(iso['r'] < r_max) & (iso['r'] > r_min)]['r']
in_mass = iso[(iso['r'] < r_max) & (iso['r'] > r_min)]['mass']

print len(in_r),len(in_mass)

tck = interpolate.splrep(in_r[::-1],in_mass[::-1])  #need to be sorted from small to large
r_new_2 = np.arange(r_min,r_max,0.01)
mass_new_2 = interpolate.splev(r_new_2,tck,der=0)
dr_new_2 = interpolate.splev(mass_new,tck,der=1)

#plot original data and spline

plt.plot(iso['mass'],iso['r'],color='b',markersize=2,marker='^')
plt.plot(mass_new,r_new,color='r',markersize=1,marker='o')  
plt.plot(mass_new_2,r_new_2,color='k',markersize=5,marker='+')
#plt.plot(mass_new,dr_new,color='g',markersize=1,marker='o')
plt.show()

#now specify a Salpeter LF

mass_arr = np.arange(0.0,2.0,0.001)

#dN_arr1 = f_salpeter(mass_arr,0.1,1.5,2.35)
#dN_arr2 = f_salpeter(mass_arr,0.1,1.5,2.00)
#dN_arr3 = f_salpeter(mass_arr,0.1,1.5,2.60)

#plt.semilogy(mass_arr,dN_arr1,label='alpha=2.35',color='red')
#plt.semilogy(mass_arr,dN_arr2,label='alpha=2.00',color='blue')
#plt.semilogy(mass_arr,dN_arr3,label='alpha=2.60',color='green')
#plt.xlabel('Mass [Msun]') ; plt.ylabel('dN')
#plt.title('Salpeter, Test Code')
#plt.show()

#now specify a Chabrier LF

dN_arr1 = f_chabrier(mass_arr,0.1,1.5,0.4,0.2)
dN_arr2 = f_chabrier(mass_arr,0.1,1.5,0.4,0.4)
dN_arr3 = f_chabrier(mass_arr,0.1,1.5,0.8,0.2)

plt.semilogy(mass_arr,dN_arr1,label='mc=.4,sig=.2',color='red',lw=3)
plt.semilogy(mass_arr,dN_arr2,label='mc=.4,sig=.4',color='blue')
plt.semilogy(mass_arr,dN_arr3,label='mc=.8,sig=.2',color='green')
plt.xlabel('Mass [Msun]') ; plt.ylabel('dN')
plt.title('Chabrier, Test Code')
plt.show()

#Now take isochorne mass-R relation, and map onto the r indices in the LF
mass_lf = interpolate.splev(lf['r'],tck,der=0)

dN_sal_luis = np.log10(f_salpeter(mass_lf,0.1,1.2,2.35))
dN_sal_darth = 10.**lf['logdN']

y = 10.**(dN_sal_luis[np.isinf(dN_sal_luis) == False])
x = lf['r'][np.isinf(dN_sal_luis) == False]

plt.plot(lf['r'],dN_sal_darth,color='r',marker='o')
plt.plot(x,8000*y,color='b',marker='o',markersize=2)
plt.xlim(r_min,r_max)
plt.ylim(0,5000)
plt.show()


