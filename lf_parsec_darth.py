import numpy as np
import matplotlib.pyplot as plt
from myanalysis import *
from mywrangle import *

sysmag2 = 'F160W'
lf1 = np.genfromtxt(open('chabrier_darth.dat','r'),comments='#',names=True,skip_header=9)
dN_darth = 10.**lf1['logdN']

lf2 = np.genfromtxt(open('chabrier_parsec.dat','r'),comments='#',names=True,skip_header=12)
dN_parsec = lf2['logdN']

max_darth = max(dN_darth)
dN_parsec = dN_parsec * max_darth / max(dN_parsec)

#now make my own "chabrier 2001" LF

error_unique = 0.001
magarr1 = np.arange(24.,31.0,0.01)
magerrarr1 = np.copy(magarr1) * 0.0 + error_unique
magarr2 = np.copy(magarr1)
magerrarr2 = np.copy(magerrarr1)

system = 'wfc3'; isoage = 14.0; isofeh = -2.0; isoafe = 0.0; sysmag1 = 'F110W'; sysmag2 = 'F160W'
mass_min_global = 0.11
start_seed = 1234
magbins_cen = np.arange(26.,31.,0.3)
magbins = magbins_cen - 0.15
maglabels = np.array([26.,27.,28.,29.,30.,31.])
masslabels = np.copy(maglabels)*0.0

phot = simulate_cmd(5000,14.0,-2.0,0.00,18.22,magarr1,magerrarr1,magarr2,magerrarr2,
            system,sysmag1,sysmag2,imftype='chabrier',mc=0.1,sigmac=0.63,mass_min=mass_min_global,testing=False,start_seed=1234)
phi_arr , maghist = np.histogram(phot[sysmag2],bins=magbins)
phi_arr_err = np.sqrt(phi_arr) 

magbins = magbins - (28.1875 - 24.6949)  #convert from STmag to Vegamag
maglabels = maglabels - (28.1875 - 24.6949) 

magbins_cen = (magbins[:-1] + magbins[1:])/2.
dmod_coma = 18.22
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.plot(lf1['F160W']+dmod_coma,dN_darth,'o',color='r',markersize=3)
ax1.plot(lf2['F160W']+dmod_coma,dN_parsec,'o',color='b',markersize=3)
ax1.errorbar(magbins_cen,phi_arr*15,yerr=phi_arr_err,color='g',marker='o',markersize=3)
ax1.set_yscale("log", nonposy='clip')
ax1.set_xlim(18,32)
ax1.set_xlabel(r'$'+sysmag2+'$ (VEGAMAG)')
ax1.set_ylabel(r'Number')
plt.show()

