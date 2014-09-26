#!/usr/bin/env python
"""This program calculates number of stars 
observable with WFC3 for Coma UFD."""

import os 
import sys
import numpy as np
import matplotlib.pyplot as plt
from myanalysis import *

#Read-in Coma Data

f = open(os.getenv('DATA')+'/HST/'+'comber_clean.cat','r')
data = np.genfromtxt(f,comments='#',names=True)

#Estimate number of * for 12 ACS pointings (202 arcsec^2 total) in fixed mag range.
datasub = data[(data['F814W'] >= 22.5) & (data['F814W'] <= 25.5)]
n_acs = len(datasub['F814W'])

#Plot Coma CMD
plt.scatter(data['F606W']-data['F814W'],data['F814W'],s=1,marker='o',color='b')
plt.scatter(datasub['F606W']-datasub['F814W'],datasub['F814W'],s=1.5,marker='o',color='r')
plt.axis([-1.5,1.0,30,20])
plt.show()

#Estimaate number of * for 4 WFC3 pointings in same mag range.
    #ACS FOV  = 202" x 202", pixel size = 0.05"
    #WFC3 FOV = 123" x 126", pixel size = 0.13" (for *IR channel*, 162"x162" for UVIS)
f_wfc3_to_acs = (4. * 123.* 126.) / (12. * 202. * 202.)

n_wfc3 = round(n_acs * f_wfc3_to_acs)

print 'N(WFC3/N(ACS): ',f_wfc3_to_acs
print 'N(ACS): ',n_acs
print 'N(WFC3): ',n_wfc3

#Map ACS magnitude range --> WFC3 magnitude range

dmod_coma = 18.22 #\pm{0.20}
mass_min_global = 0.2  #dummy value
ndum, mass_min_acs, mass_max_acs = estimate_required_n(1000,14.0,-2.5,0.4,'acs','F814W',dmod_coma,22.5,25.5,  
                            imftype='salpeter',alpha=2.35,mass_min_global=mass_min_global)

print 'ACS Mass Range: ',mass_min_acs,' < M < ',mass_max_acs

system = 'wfc3'; isoage = 14.0; isofeh = -2.5; isoafe = 0.4; sysmag1 = 'F110W'; sysmag2 = 'F160W'

iso0 = read_iso_darth(isoage,isofeh,isoafe,system)
w1 = np.argmin(abs(iso0['mass']-mass_max_acs))
w2 = np.argmin(abs(iso0['mass']-mass_min_acs))
mag_min_wfc3 = iso0[sysmag2][w1] + dmod_coma
mag_max_wfc3 = iso0[sysmag2][w2] + dmod_coma

print 'M_max = ',mass_max_acs,' ==> ',22.5,' (ACS) --> ',mag_min_wfc3,' (WFC3)'
print 'M_min = ',mass_min_acs,' ==> ',25.5,' (ACS) --> ',mag_max_wfc3,' (WFC3)'

#Extrapolate number of stars for three different IMFs in WFC3

#IMF 1: power law, alpha = 1.3
#IMF 2: lognormal, Mc = 0.3
#IMF 3: lognormal, Mc = 0.4

mass_min_global = 0.11 #set to M << range (M (data))

mag_max_prop = 26.3  #in F160W, 26.9 in F110W

nstars = n_wfc3  #This is the number of stars we want in the limited range observed by ACS.

#nall is the estimated number required down to the lower mass limit of the isochrone (Mglobal =~ 0.11 Msun)

nall, mass_min_fit, mass_max_fit = estimate_required_n(nstars,14.0,-2.5,0.4,'acs','F814W',dmod_coma,mag_min_wfc3,mag_max_wfc3,  
                            imftype='salpeter',alpha=1.3,mass_min_global=mass_min_global)

seed_arr = 5*np.arange(20) + 101

#define a magnitude error array, guided on analysis from Herc data for ACS?, or on a constant value

error_unique = 0.02
magarr1 = np.arange(24.,30.,0.01)
magerrarr1 = np.copy(magarr1) * 0.0 + error_unique
magarr2 = np.copy(magarr1)
magerrarr2 = np.copy(magerrarr1)

for iseed,seed in enumerate(seed_arr):

    phot1 = simulate_cmd(nall,isoage,isofeh,isoafe,dmod_coma,magarr1,magerrarr1,magarr2,magerrarr2,
            system,sysmag1,sysmag2,imftype='salpeter',alpha=1.3,mass_min=mass_min_global,start_seed=seed)  #1.5 bc of additional
    phot1 = filter_phot(phot1,system,sysmag1,sysmag2,x1=-2.0,x2=2.5,y1=mag_min_wfc3,y2=mag_max_prop)
 
    print iseed,len(phot1[(phot1[sysmag1] >= mag_min_wfc3) & (phot1[sysmag2] <= mag_max_wfc3)]),len(phot1[(phot1[sysmag1] >= mag_min_wfc3) & (phot1[sysmag2] <= mag_max_prop)])


#nall, mass_min_fit, mass_max_fit = estimate_required_n(nstars,14.0,-2.5,0.4,'acs','F814W',dmod_coma,mag_min_wfc3,mag_max_wfc3,  
#                            imftype='chabrier',mc=param_in,sigmac=sigmac,mass_min_global=mass_min_global)
#phot2 = simulate_cmd(nall,isoage,isofeh,isoafe,dmod0,magarr1,magerrarr1,magarr2,magerrarr2,
#                            system,sysmag1,sysmag2,imftype='chabrier',mc=0.3,sigmac=0.69,mass_min=mass_min_global,start_seed=start_seed)  #1.5 bc of additional

#nall, mass_min_fit, mass_max_fit = estimate_required_n(nstars,14.0,-2.5,0.4,'acs','F814W',dmod_coma,mag_min_wfc3,mag_max_wfc3,  
#                            imftype='chabrier',alpha1=param_in,alpha2=alpha2,mass_min_global=mass_min_global)
#phot3 = simulate_cmd(nall,isoage,isofeh,isoafe,dmod0,magarr1,magerrarr1,magarr2,magerrarr2,
#                            system,sysmag1,sysmag2,imftype='chabrier',mc=0.4,sigmac=0.69,mass_min=mass_min_global,start_seed=start_seed)  #1.5 bc of additional

#Now filter the isochrones by the output magnitude limits of the proposal; note no constraints on sysmag1 at this point, but should be folded in

#phot2     = filter_phot(phot1,system,sysmag1,sysmag2,x1=-1.5,x2=2.5,y1=mag_min_wfc3,y2=mag_max_prop)  #make broad enough for mock data 
#phot3     = filter_phot(phot1,system,sysmag1,sysmag2,x1=-1.5,x2=2.5,y1=mag_min_wfc3,y2=mag_max_prop)  #make broad enough for mock data 

#Plot results

